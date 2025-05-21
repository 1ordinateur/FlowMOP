"""
TXT Flow Data Converter

Handles conversion of flow cytometry text files to Parquet format with sparse column support.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gc
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TXTConverter:
    """Converts flow cytometry text files to Parquet format with sparse marker support."""
    
    def __init__(self, base_dir: str, output_dir: str, metadata_path: str = None):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path) if metadata_path else self.output_dir / "metadata.parquet"
        self.all_markers = set()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _standardize_marker_name(self, name: str) -> str:
        """Normalize marker names to lowercase with underscores"""
        return re.sub(r'\W+', '_', name.lower()).strip('_')

    def process_files(self):
        """Process files with true streaming writes using PyArrow"""
        try:
            txt_files = list(self.base_dir.rglob("*.txt"))
            if not txt_files:
                raise ValueError(f"No TXT files found in {self.base_dir}")

            parquet_path = self.output_dir / "events.parquet"
            metadata_records = []
            total_markers = set()
            
            # Initialize writer with dynamic schema
            writer = None
            schema = None

            for file_idx, file_path in enumerate(txt_files):
                try:
                    file_markers = set()
                    # Generate consistent sample ID from relative path
                    relative_path = file_path.relative_to(self.base_dir)
                    # Remove extensions and normalize separators
                    base_name = str(relative_path).split('.', 1)[0]  # Split on first dot
                    sample_id = re.sub(r'[^a-zA-Z0-9]', '_', base_name)  # Replace special chars
                    
                    for chunk_idx, df_chunk in enumerate(pd.read_csv(file_path, sep='\t', chunksize=100000)):
                        # Standardize column names
                        df_chunk.columns = [self._standardize_marker_name(col) for col in df_chunk.columns]
                        
                        # Add identifiers
                        df_chunk['sample_id'] = sample_id
                        df_chunk['event_id'] = np.arange(len(df_chunk)) + (chunk_idx * 100000)
                        
                        # Track markers
                        file_markers.update(df_chunk.columns)
                        total_markers.update(df_chunk.columns)
                        
                        # Reindex to current schema
                        df_chunk = df_chunk.reindex(columns=list(total_markers), fill_value=np.nan)
                        
                        # Update schema if needed
                        current_schema = pa.Schema.from_pandas(df_chunk)
                        
                        if not writer:
                            # Initialize writer with first chunk's schema
                            writer = pq.ParquetWriter(
                                parquet_path,
                                current_schema,
                                compression='snappy',
                                version='2.6'
                            )
                        elif current_schema != writer.schema:
                            # Close existing writer and create new one with updated schema
                            writer.close()
                            writer = pq.ParquetWriter(
                                parquet_path,
                                current_schema,
                                compression='snappy',
                                version='2.6'
                            )
                        
                        # Write chunk
                        table = pa.Table.from_pandas(df_chunk, schema=current_schema)
                        writer.write_table(table)
                        
                        # Explicit memory cleanup
                        del df_chunk, table
                        gc.collect()
                    
                    # Record metadata
                    metadata_records.append({
                        'sample_id': sample_id,
                        'file_path': str(file_path),
                        'relative_path': str(relative_path),
                        'markers': list(file_markers),
                        'cohort': str(file_path.parent.name)
                    })
                    
                    logger.info(f"Processed {file_path.name} ({file_idx+1}/{len(txt_files)})")

                except Exception as e:
                    logger.error(f"Skipped {file_path}: {str(e)}")
                    continue
                    
            # Finalize outputs
            if writer:
                writer.close()
            
            pd.DataFrame(metadata_records).to_parquet(
                self.metadata_path,
                engine='pyarrow', 
                compression='snappy'
            )

            logger.info(f"Completed processing {len(txt_files)} files. Total markers: {len(total_markers)}")

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    def _create_metadata(self, file_path: Path, markers: set) -> Dict:
        """Generate metadata record for a file"""
        return {
            'file_path': str(file_path),
            'markers_present': list(markers),
            'cohort': file_path.parent.name,
            'processing_time': pd.Timestamp.now()
        }

if __name__ == "__main__":
    converter = TXTConverter(
        base_dir="/path/to/txt/files",
        output_dir="/path/to/output"
    )
    converter.process_files() 