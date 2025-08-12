# Check how many images are there for MH012 (both years)
from pathlib import Path

path_2021 = Path(r"C:\Users\Lalit Hire\OneDrive\Desktop\APE_07\data\cropped_data\Maharashtra\MH012\2021")  # Replace with correct full path
path_2022 = Path(r"CC:\Users\Lalit Hire\OneDrive\Desktop\APE_07\data\cropped_data\Maharashtra\MH012\2022")

images_2021 = list(path_2021.glob("*.tif"))
images_2022 = list(path_2022.glob("*.tif"))

print(f"📂 MH012 - 2021: {len(images_2021)} images")
print(f"📂 MH012 - 2022: {len(images_2022)} images")