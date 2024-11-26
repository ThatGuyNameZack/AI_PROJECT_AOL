import os

base_dir = '/Users/rafisatria/Documents/GitHub/AI_PROJECT_AOL/Student-engagement-dataset/'
engaged_dir = os.path.join(base_dir, 'engaged')
not_engaged_dir = os.path.join(base_dir, 'not engaged')

print(f"Engaged directory exists: {os.path.exists(engaged_dir)}")
print(f"Not engaged directory exists: {os.path.exists(not_engaged_dir)}")

print(f"Files in engaged directory: {os.listdir(engaged_dir) if os.path.exists(engaged_dir) else 'Folder not found'}")
print(f"Files in not engaged directory: {os.listdir(not_engaged_dir) if os.path.exists(not_engaged_dir) else 'Folder not found'}")
