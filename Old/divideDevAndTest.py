import os

i = 0
for file in os.listdir("Test"):
    if i%2 == 1:
        os.remove(os.path.join("Test", file))
    i += 1