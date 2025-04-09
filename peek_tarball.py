import tarfile
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def peek_tarball(file_path: str, list_only: bool = True):
    """Peek inside a tarball.
    
    Args:
        file_path: Path to .tar.gz file
        list_only: If True, just list contents. If False, extract.
    """
    try:
        logger.info(f"Opening {file_path}")
        with tarfile.open(file_path, 'r:gz') as tar:
            # List all contents
            logger.info("Contents:")
            for member in tar.getmembers()[:5]:  # Show first 5 files
                logger.info(f"{member.name} ({member.size} bytes)")
            
            total_files = len(tar.getmembers())
            logger.info(f"Total files: {total_files}")
            
            # Extract a sample if requested
            if not list_only:
                sample_file = tar.getmembers()[0]
                logger.info(f"Extracting sample file: {sample_file.name}")
                tar.extract(sample_file, "sample_extract")
    
    except Exception as e:
        logger.error(f"Error processing tarball: {str(e)}")

if __name__ == "__main__":
    # Check if file exists
    file_path = sys.argv[1] if len(sys.argv) > 1 else "Berlin.tar.gz.ab"
    
    if not Path(file_path).exists():
        logger.error(f"File {file_path} not found")
        sys.exit(1)
        
    peek_tarball(file_path)