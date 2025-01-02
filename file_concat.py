import os
from pathlib import Path

def concatenate_files_recursive(directory_path, output_file='concatenated_output.txt'):
    """
    Recursively concatenates all text files in a directory and its subdirectories with specific formatting.

    Args:
        directory_path (str): Path to the root directory containing files
        output_file (str): Name of the output file
    """
    # Convert to Path object for better path handling
    root_directory = Path(directory_path)

    # Ensure directory exists
    if not root_directory.is_dir():
        raise ValueError(f"'{directory_path}' is not a valid directory")

    # Open output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Recursively iterate through all files in directory and subdirectories
        for file_path in root_directory.rglob('*'):
            if file_path.is_file():
                try:
                    # Get relative path from root directory
                    relative_path = file_path.relative_to(root_directory)

                    # Read content of each file
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()

                    # Write formatted content to output file
                    outfile.write(f"```{relative_path}\n")
                    outfile.write(content)
                    if not content.endswith('\n'):
                        outfile.write('\n')
                    outfile.write("'''\n\n\n")
                except UnicodeDecodeError:
                    print(f"Skipping {relative_path}: Not a text file")
                except Exception as e:
                    print(f"Error processing {relative_path}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'concatenated_output.txt'
        try:
            concatenate_files_recursive(directory_path, output_file)
            print(f"Files successfully concatenated to {output_file}")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Usage: python script.py <directory_path> [output_file]")
