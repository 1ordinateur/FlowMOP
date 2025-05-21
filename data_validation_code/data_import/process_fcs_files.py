"""
FCS Processing Script

This script processes FCS files using the FCSConverter class and integrates
additional metadata from an Excel sheet.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd
from dask.distributed import Client, LocalCluster
import numpy as np

from fcs_to_parquet import FCSConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FCSProcessor:
    """Processes FCS files with additional metadata from Excel."""
    
    def __init__(
        self,
        base_dir: str,
        metadata_excel: str,
        output_dir: str,
        filename_column: str = "filename",
        output_name: str = "processed_data",
        n_workers: int = 4,
        memory_limit: str = "4GB",
        chunk_size: int = 100000
    ):
        """
        Initialize the FCS processor.

        Args:
            base_dir: Base directory containing FCS files (can have subdirectories)
            metadata_excel: Path to Excel file containing metadata
            output_dir: Directory for output parquet files
            filename_column: Name of the column in Excel that contains FCS filenames
            output_name: Base name for output files (without extension)
            n_workers: Number of Dask workers
            memory_limit: Memory limit per worker
            chunk_size: Number of events to process at once
        """
        self.base_dir = Path(base_dir)
        self.metadata_excel = Path(metadata_excel)
        self.output_dir = Path(output_dir)
        self.filename_column = filename_column
        self.output_name = output_name
        self.n_workers = n_workers
        self.memory_limit = memory_limit
        self.chunk_size = chunk_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.output_dir / f"{self.output_name}_metadata.parquet"
        
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load and validate metadata from Excel, CSV or Parquet file.
        
        Returns:
            DataFrame containing metadata
        """
        logger.info(f"Loading metadata from {self.metadata_excel}")
        
        try:
            # Try reading based on file extension
            if self.metadata_excel.suffix == '.parquet':
                metadata_df = pd.read_parquet(self.metadata_excel)
            else:
                # Try reading as CSV first, if that fails try Excel
                try:
                    metadata_df = pd.read_csv(self.metadata_excel, delimiter=';')
                except Exception:
                    metadata_df = pd.read_excel(self.metadata_excel, engine='openpyxl')
            logger.info(f"Available columns in metadata: {metadata_df.columns.tolist()}")
            
            if self.filename_column not in metadata_df.columns:
                # Try to find similar column names
                similar_cols = [col for col in metadata_df.columns 
                              if any(term in col.lower() 
                                    for term in ['file', 'name', 'fcs'])]
                
                error_msg = (
                    f"Column '{self.filename_column}' not found in metadata file. "
                    f"Available columns are: {metadata_df.columns.tolist()}"
                )
                if similar_cols:
                    error_msg += f"\nPossible filename columns: {similar_cols}"
                raise ValueError(error_msg)
                
            # Rename the column to 'filename' for consistency
            metadata_df = metadata_df.rename(columns={self.filename_column: 'filename'})
            
            # Clean filenames in Excel metadata to match FCS processing
            metadata_df['filename'] = metadata_df['filename'].apply(
                lambda x: re.sub(r'[^a-zA-Z0-9_.]', '_', str(x)).lower().strip('_')
            )
            
            return metadata_df
            
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise
            
    def _validate_files(self, metadata_df: pd.DataFrame) -> Tuple[Dict[str, Path], pd.DataFrame]:
        """
        Validate files in metadata and create filename mapping, skipping missing files.
        
        Args:
            metadata_df: DataFrame containing metadata
            
        Returns:
            Tuple of (file mapping dict, filtered metadata DataFrame)
        """
        logger.info("Validating FCS files...")
        
        # Find all FCS files and clean their names
        all_files = list(self.base_dir.rglob("*.fcs"))
        file_map = {
            re.sub(r'[^a-zA-Z0-9_.]', '_', f.name).lower().strip('_'): f
            for f in all_files
        }
        
        logger.info(f"Found {len(file_map)} FCS files in directory")
        
        # Track missing files
        missing_files = []
        valid_files = []
        
        for filename in metadata_df['filename']:
            if filename not in file_map:
                missing_files.append(filename)
            else:
                valid_files.append(filename)
        
        if missing_files:
            logger.warning(
                f"Skipping {len(missing_files)} files not found in directory: "
                f"{missing_files[:10]}"
                f"{' and more...' if len(missing_files) > 10 else ''}"
            )
        
        # Filter metadata to only include files that exist
        filtered_metadata = metadata_df[metadata_df['filename'].isin(valid_files)].copy()
        logger.info(f"Processing {len(filtered_metadata)} files with valid metadata")
        
        return file_map, filtered_metadata
        
    def process(self):
        """Processing without error handling"""
        # Remove outer try/except block
        excel_metadata = self._load_metadata()
        
        with LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=1,
            memory_limit=self.memory_limit
        ) as cluster, Client(cluster) as client:
            logger.info(f"Dask dashboard: {client.dashboard_link}")
            
            converter = FCSConverter(
                input_path=self.base_dir,
                output_dir=self.output_dir,
                metadata_path=self.metadata_path,
                chunk_size=self.chunk_size
            )
            converter.convert(client=client)
            
            fcs_metadata = pd.read_parquet(self.metadata_path)
            
            # Add explicit column check
            missing_cols = {'filename', 'sample_id', 'acquisition_date'} - set(fcs_metadata.columns)
            if missing_cols:
                raise KeyError(f"Critical columns missing: {missing_cols}")
            
            merged_metadata = pd.merge(
                fcs_metadata,
                excel_metadata,
                left_on=['filename', 'tube_id'],
                right_on=[self.filename_column, 'tube_id'],
                how='left',
                validate='one_to_one'
            )
            
            # Add barcode validation status
            merged_metadata['barcode_status'] = np.where(
                merged_metadata['barcode'].notna(),
                'valid' if self._validate_barcode(merged_metadata['barcode']) else 'invalid',
                'missing'
            )
            
            # Ensure all paths are strings
            merged_metadata['file_path'] = merged_metadata['file_path'].astype(str)
            merged_metadata['filename'] = merged_metadata['filename'].astype(str)
            
            # Convert all potential path columns
            path_columns = ['file_path', 'filename', 'relative_path', 'full_path']
            for col in path_columns:
                if col in merged_metadata:
                    merged_metadata[col] = merged_metadata[col].astype(str)
            
            # Before saving
            logger.info("Merged metadata columns:")
            logger.info(merged_metadata.dtypes)
            
            # Check for path-like objects
            for col in merged_metadata.columns:
                if merged_metadata[col].apply(lambda x: isinstance(x, Path)).any():
                    logger.warning(f"Column {col} contains Path objects - converting to strings")
                    merged_metadata[col] = merged_metadata[col].astype(str)
            
            # Save final metadata
            merged_metadata.to_parquet(
                self.metadata_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            logger.info("Processing completed successfully")
            logger.info(f"Data written to: {self.output_dir / f'{self.output_name}.parquet'}")
            logger.info(f"Metadata written to: {self.metadata_path}")

    def _validate_barcode(self, barcode_series: pd.Series) -> bool:
        """Validate barcode format using Luhn algorithm"""
        def luhn_check(code: str) -> bool:
            if not code.isdigit():
                return False
            digits = list(map(int, code))
            checksum = digits[-1]
            total = sum(d * (2 - i % 2) for i, d in enumerate(digits[:-1]))
            return (total + checksum) % 10 == 0
        
        return barcode_series.apply(luhn_check).all()

class TXTProcessor(FCSProcessor):
    """Processes TXT flow files with directory structure support."""
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata with path handling."""
        metadata_df = super()._load_metadata()
        
        # Add full paths
        metadata_df['full_path'] = metadata_df['filename'].apply(
            lambda x: self.base_dir / x
        )
        return metadata_df
        
    def _validate_files(self, metadata_df: pd.DataFrame) -> Tuple[Dict[str, Path], pd.DataFrame]:
        """Validate files using path from metadata with subdirectory support."""
        logger.info("Validating TXT files with subdirectory paths...")
        
        # Construct full paths preserving directory structure
        metadata_df['full_path'] = metadata_df['filename'].apply(
            lambda x: self.base_dir / x
        )
        
        # Check existence of constructed paths
        metadata_df['file_exists'] = metadata_df['full_path'].apply(lambda x: x.exists())
        missing_files = metadata_df[~metadata_df['file_exists']]
        
        if not missing_files.empty:
            logger.warning(f"Missing {len(missing_files)} files. Examples:\n%s", 
                          missing_files['filename'].head(5).to_string(index=False))
            
        valid_metadata = metadata_df[metadata_df['file_exists']].copy()
        
        # Create mapping from filename to full path
        file_map = {
            row['filename']: row['full_path']
            for _, row in valid_metadata.iterrows()
        }
        
        return file_map, valid_metadata
        
    def process(self):
        """TXT-specific processing pipeline."""
        logger.info("Starting TXT processing")
        from txt_to_parquet import TXTConverter
        try:
            # Load metadata
            excel_metadata = self._load_metadata()
            
            # Validate files
            file_map, filtered_metadata = self._validate_files(excel_metadata)
            
            # Convert files
            converter = TXTConverter(
                base_dir=self.base_dir,
                output_dir=self.output_dir,
                metadata_path=self.metadata_path  # Pass explicit metadata path
            )
            
            converter.process_files()
            
            # Merge metadata
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
                
            fcs_metadata = pd.read_parquet(self.metadata_path)
            print(fcs_metadata["file_path"])
            print(filtered_metadata["filename"])
            
            # Add sample_id to filtered metadata
            filtered_metadata['sample_id'] = filtered_metadata['filename'].apply(
                lambda x: re.sub(r'[^a-zA-Z0-9]', '_', x.split('.', 1)[0])
            )
            
            merged_metadata = pd.merge(
                fcs_metadata,
                filtered_metadata,
                on='sample_id',  # Now using the consistent ID
                how='left'
            )
            
            # Ensure all paths are strings
            merged_metadata['file_path'] = merged_metadata['file_path'].astype(str)
            merged_metadata['filename'] = merged_metadata['filename'].astype(str)
            
            # Convert all potential path columns
            path_columns = ['file_path', 'filename', 'relative_path', 'full_path']
            for col in path_columns:
                if col in merged_metadata:
                    merged_metadata[col] = merged_metadata[col].astype(str)
            
            # Before saving
            logger.info("Merged metadata columns:")
            logger.info(merged_metadata.dtypes)
            
            # Check for path-like objects
            for col in merged_metadata.columns:
                if merged_metadata[col].apply(lambda x: isinstance(x, Path)).any():
                    logger.warning(f"Column {col} contains Path objects - converting to strings")
                    merged_metadata[col] = merged_metadata[col].astype(str)
            
            # Save final metadata to same path
            merged_metadata.to_parquet(
                self.metadata_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            logger.info("Processing completed successfully")

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

def main():
    """Main function to run the processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process flow cytometry files with metadata')
    parser.add_argument('--base-dir', required=True, help='Base directory containing data files')
    parser.add_argument('--metadata', required=True, help='Path to metadata Excel file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--filename-column', default='filename', 
                       help='Column in Excel with filenames')
    parser.add_argument('--output-name', default='processed_data', 
                       help='Base name for output files')
    parser.add_argument('--workers', type=int, default=4, 
                       help='Number of Dask workers')
    parser.add_argument('--memory-limit', default='4GB', 
                       help='Memory limit per worker')
    parser.add_argument('--chunk-size', type=int, default=100000, 
                       help='Processing chunk size')
    # Add processor type argument
    parser.add_argument('--processor-type', choices=['fcs', 'txt'], default='fcs',
                       help='Type of processor to use (fcs or txt)')

    args = parser.parse_args()
    
    # Choose processor based on type
    if args.processor_type == 'txt':
        from txt_to_parquet import TXTConverter
        processor_cls = TXTProcessor
    else:
        processor_cls = FCSProcessor
    
    # Initialize processor
    processor = processor_cls(
        base_dir=args.base_dir,
        metadata_excel=args.metadata,
        output_dir=args.output_dir,
        filename_column=args.filename_column,
        output_name=args.output_name,
        n_workers=args.workers,
        memory_limit=args.memory_limit,
        chunk_size=args.chunk_size
    )
    
    processor.process()

if __name__ == "__main__":
    main()