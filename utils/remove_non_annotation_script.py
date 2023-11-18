import os

file_names = os.listdir(".")
for file_name in file_names:
    extension = file_name.split(".")[-1]
    name = file_name[: len(file_name) - len(extension)-1]
    if extension == "png":
        if not os.path.exists(name + ".json"):
            os.remove(file_name)
