import os
def check_path_exits(path):
    if os.path.exists(path):
        return True
    print(f"Data path doesn't exits, Path: '{path}'")
    return False