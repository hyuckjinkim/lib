"""
모든 경로에서 ._ 파일을 삭제한다.

Example: ```python lib/delete_macos_dotfiles.py```
"""

if __name__=="__main__":
    import os,sys
    sys.path.append(os.path.abspath(''))

    from lib.python.filesystem_utils import delete_files_with_pattern
    delete_files_with_pattern(pattern='./**/._*', return_file_status=True)