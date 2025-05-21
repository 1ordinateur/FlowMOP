"""
FCS/LMD to Parquet Converter with sparse marker handling
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import warnings
import uuid
import json
from datetime import datetime
import re
import hashlib

import dask.array as da
import dask.dataframe as dd
import dask
import numpy as np
from dask.distributed import Client
import pandas as pd
import pyarrow as pa
from fcsparser import parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FCSConverter:
    def __init__(
        self,
        input_path: str,
        output_dir: str,
        case_info_paths: Optional[List[str]] = None,
        chunk_size: int = 100000,
        barcode_mapping: Optional[Dict[str, str]] = None,
        metadata_path: Optional[str] = None
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.case_info_paths = [Path(p) for p in case_info_paths] if case_info_paths else []
        self.chunk_size = chunk_size
        self.barcode_mapping = barcode_mapping or {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track markers by panel
        self.panel_markers: Dict[str, Set[str]] = {}
        
        # Load case info files
        self.case_info = self._load_case_info()
        self.metadata_path = metadata_path

    def _load_case_info(self) -> Dict:
        """Load and merge all case info JSON files"""
        case_info = {}
        for path in self.case_info_paths:
            try:
                with open(path) as f:
                    data = json.load(f)
                    # Extract panel from path
                    panel = re.search(r'(MLL[59]F|Bonn|Erlangen|Berlin)', str(path)).group(1)
                    case_info[panel] = data
                logger.info(f"Loaded case info from {path}")
            except Exception as e:
                logger.error(f"Error loading case info from {path}: {str(e)}")
        return case_info

    def _standardize_marker_name(self, name: str) -> str:
        """Normalize marker names to a standard format"""
        # Remove special characters and standardize
        clean_name = re.sub(r'[^\w\s-]', '', name)
        # Convert to lowercase and replace spaces/hyphens with underscores
        return re.sub(r'[-\s]+', '_', clean_name).lower().strip('_')

    def _process_events(self, parsed_data, sample_id: str, panel: str) -> dd.DataFrame:
        """Process events using fcsparser output"""
        # Access data and metadata
        df = parsed_data[1]  # DataFrame of events
        meta = parsed_data[0]  # Metadata dict
        
        # Get standardized channel names
        channels = [self._standardize_marker_name(ch) for ch in df.columns]
        
        # Update panel markers
        if panel not in self.panel_markers:
            self.panel_markers[panel] = set()
        self.panel_markers[panel].update(channels)
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(df, chunksize=self.chunk_size)
        ddf.columns = channels  # Apply standardized names
        
        # Add identifiers - FIXED VERSION
        ddf = ddf.assign(
            sample_id=lambda _: sample_id,
            panel=lambda _: panel,
            event_number=ddf.index
        )
        
        return ddf

    def _extract_metadata(
        self, 
        fcs_file: Path, 
        meta: dict,  # Direct fcsparser metadata dict
        sample_id: str,
        panel: str
    ) -> Dict:
        """Metadata extraction using fcsparser's native structure"""
        # Extract pure filename without path
        raw_filename = fcs_file.name
        
        # Clean special characters and normalize
        clean_filename = re.sub(r'[^a-zA-Z0-9_.]', '_', raw_filename).lower().strip('_')
        
        # Get acquisition date with fallbacks
        try:
            acq_date = datetime.strptime(
                f"{meta['date']} {meta['time']}",
                '%Y-%b-%d %H:%M:%S'  # fcsparser's standardized format
            )
        except (KeyError, ValueError):
            acq_date = datetime.fromtimestamp(fcs_file.stat().st_mtime)
        
        # Get standardized marker names from dataframe columns
        std_markers = [self._standardize_marker_name(m) for m in meta['_channels_']]
        
        return {
            'file_path': str(fcs_file),
            'filename': clean_filename,
            'sample_id': sample_id,
            'panel': panel,
            'acquisition_date': acq_date,
            'total_events': meta['$TOT'],
            'markers': ','.join(std_markers),
            'cytometer': meta.get('CYT', 'unknown'),
            'software': meta.get('LAST_MODIFIED', {}).get('SOFTWARE', 'unknown'),
            'is_validated': False
        }

    def _combine_parquets(self, parquet_paths: List[str], output_path: Path, chunk_size: int = 10):
        """Combine multiple parquet files into one"""
        logger.info(f"Combining {len(parquet_paths)} parquet files...")
        
        try:
            for i in range(0, len(parquet_paths), chunk_size):
                chunk = parquet_paths[i:i + chunk_size]
                dfs = [dd.read_parquet(p) for p in chunk]
                
                if not dfs:
                    continue
                    
                combined = dd.concat(dfs)
                combined.to_parquet(
                    output_path,
                    engine='pyarrow',
                    compression='snappy',
                    write_index=False,
                    append=(i > 0)
                )
                
                # Clear memory
                del dfs
                del combined
                
        except Exception as e:
            logger.error(f"Error combining parquets: {str(e)}")
            raise

    def _get_sample_id(self, file_path: Path) -> str:
        """Generate consistent 40-character hash from file metadata"""
        file_stat = file_path.stat()
        hash_input = f"{file_path.absolute()}-{file_stat.st_size}-{file_stat.st_mtime_ns}"
        return hashlib.md5(hash_input.encode()).hexdigest()  # Remove the [:12] slice

    def convert(self, client: Optional[Client] = None):
        """Convert FCS files to parquet with parallel processing"""
        logger.info("Starting conversion process...")
        client = client or Client(
            n_workers=16,
            threads_per_worker=1,
            memory_limit='8GB'
        )
        
        # Setup paths and find files
        files = list(self.input_path.rglob("*.LMD")) + list(self.input_path.rglob("*.fcs"))
        intermediate_dir = self.output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing parquets PROPERLY
        existing_parquets = {}
        if intermediate_dir.exists():
            for p in intermediate_dir.glob("*.parquet"):
                # Match full 40-character hash at start of filename
                if match := re.match(r"^([a-f0-9]{40})", p.stem):  # Changed from 12 to 40
                    existing_parquets[match.group(1)] = p
                    
        # Filter out files that have already been processed
        files_to_process = []
        for f in files:
            # Generate full 40-character hash for comparison
            sample_id = self._get_sample_id(f)
            if sample_id not in existing_parquets:
                files_to_process.append((f, sample_id))
            else:
                parquet_paths.append(str(existing_parquets[sample_id]))
                
        logger.info(f"Found {len(files)} files, {len(files_to_process)} need processing")
        
        if not files_to_process:
            logger.info("All files already processed, skipping to combination step")
            parquet_paths = [str(p) for p in intermediate_dir.glob("*.parquet")]
        else:
            # First get schema from all files
            logger.info("Collecting marker information...")
            encoding_warnings = 0
            
            def _get_file_schema(file_path: Path) -> Set[str]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("always", category=UnicodeWarning)
                    meta = parse(file_path, reformat_meta=True, meta_data_only=True)
                    if len(w) > 0:
                        nonlocal encoding_warnings
                        encoding_warnings += 1
                return set(self._standardize_marker_name(ch) for ch in meta['_channels_'])
                
            schema_results = dask.compute(*[
                dask.delayed(_get_file_schema)(f) for f, _ in files_to_process
            ])
            
            # Build schema
            required_columns = {'sample_id', 'panel', 'event_number'}
            all_markers = required_columns.union(*schema_results)
            column_order = sorted(all_markers)
            
            logger.info(f"Found {len(column_order)} total columns")
            if encoding_warnings > 0:
                logger.info(f"Note: {encoding_warnings} files had UTF-8 encoding warnings (this is normal)")
            
            # Process files in batches without client restarts
            batch_size = 1000  # Adjust based on file sizes
            parquet_paths = [str(p) for p in intermediate_dir.glob("*.parquet")]  # Include existing parquets
            metadata_list = []
            
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {len(files_to_process)//batch_size + 1}")
                
                # Process batch
                batch_results = dask.compute(*[
                    dask.delayed(self._process_single_file)(f, sid, column_order) 
                    for f, sid in batch
                ])
                
                # Collect results
                for result in batch_results:
                    if result['success']:
                        metadata_list.append(result['metadata'])
                        parquet_paths.append(result['parquet_path'])
                
                # Clear task state
                client.cancel(list(client.futures))
        
        # Save metadata
        pd.DataFrame(metadata_list).to_parquet(
            self.metadata_path or (self.output_dir / "metadata.parquet"),
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Combine parquets in smaller chunks
        logger.info("Combining parquets...")
        final_path = self.output_dir / "events.parquet"
        
        chunk_size = 100
        for i in range(0, len(parquet_paths), chunk_size):
            chunk = parquet_paths[i:i + chunk_size]
            logger.info(f"Combining chunk {i//chunk_size + 1} of {len(parquet_paths)//chunk_size + 1}")
            
            # Read each parquet file explicitly
            dfs = [dd.read_parquet(p) for p in chunk]
            combined = dd.concat(dfs)
            
            # Write with append mode
            combined.to_parquet(
                final_path,
                engine='pyarrow',
                compression='snappy',
                write_index=False,
                append=(i > 0),
                ignore_divisions=True
            )
            
            # Clear memory
            del dfs, combined
            client.cancel(list(client.futures))  # Cancel any pending futures to clear memory
        
        # Cleanup
        try:
            import shutil
            shutil.rmtree(intermediate_dir)
            logger.info("Cleaned up intermediate files")
        except Exception as e:
            logger.warning(f"Could not clean up intermediate files: {str(e)}")
        
        logger.info(f"Conversion complete. Files saved to {self.output_dir}")

    def _get_panel_from_path(self, file_path: Path) -> str:
        """Extract panel information from file path"""
        panel_match = re.search(r'(MLL[59]F|Bonn|Erlangen|Berlin)', str(file_path))
        return panel_match.group(1) if panel_match else 'unknown'

    def _process_single_file(self, file_path: Path, sample_id: str, column_order: List[str]) -> Dict[str, Any]:
        """Process a single FCS file to parquet"""
        try:
            # Parse file
            parsed = parse(file_path, reformat_meta=True)
            panel = self._get_panel_from_path(file_path)
            
            # Create DataFrame and standardize columns
            df = dd.from_pandas(
                parsed[1],
                npartitions=1  # Single partition since we're processing one file at a time
            )
            df.columns = [self._standardize_marker_name(ch) for ch in df.columns]
            
            # Add required columns
            df = df.assign(
                sample_id=sample_id,
                panel=panel,
                event_number=da.arange(len(df))  # Removed chunks parameter since we're using npartitions
            )
            
            # Add missing columns with NaN
            for col in set(column_order) - set(df.columns):
                df[col] = np.nan
            
            # Ensure consistent column order
            df = df[column_order]
            
            # Save to parquet as single file
            output_path = self.output_dir / "intermediate" / f"{sample_id}.parquet"
            df.to_parquet(
                output_path,
                engine='pyarrow',
                compression='snappy',
                write_index=False,
                # Force single file output
                name_function=lambda i: output_path.name,
                version='1.0'  # Use parquet version 1.0 for single file
            )
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, parsed[0], sample_id, panel)
            
            return {
                'success': True,
                'parquet_path': str(output_path),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                'success': False,
                'parquet_path': None,
                'metadata': None
            }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert FCS/LMD files to Parquet format')
    parser.add_argument('--input-path', required=True, help='Input directory containing FCS/LMD files')
    parser.add_argument('--output-dir', required=True, help='Output directory for parquet files')
    parser.add_argument('--case-info', nargs='*', help='Paths to case info JSON files')
    parser.add_argument('--chunk-size', type=int, default=100000, help='Chunk size for processing')
    
    args = parser.parse_args()
    
    converter = FCSConverter(
        input_path=args.input_path,
        output_dir=args.output_dir,
        case_info_paths=args.case_info,
        chunk_size=args.chunk_size
    )
    
    converter.convert()

if __name__ == "__main__":
    main()
