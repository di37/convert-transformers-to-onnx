import os

dirs = [
    'saved_models',
    'src'
]

for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f:
        pass

files = [
    ".gitignore",
    "main.py",
    os.path.join("src", "__init__.py"),
    os.path.join("src", "bertBaseMultiClass.py"),
    os.path.join("src", "sentencesSmilarity.py"),
]

for file_ in files:
    with open(file_, "w") as f:
        pass
    