import os
import sys
import joblib



def load_joblib_with_remapping(file_path):
    try:
        # Load the model with `joblib.load` in `mmap_mode`
        model = joblib.load(file_path, mmap_mode='r')
        print("File loaded successfully using joblib with mmap_mode.")
        return model
    except ModuleNotFoundError as e:
        print(f"ModuleNotFoundError: {e}")
    except Exception as e:
        print(f"General loading error with joblib: {e}")
sys.path.append(r'C:\Users\gabri\Desktop\amsti_algo\Amsti_Repo\wpi_onderzoekswaardigheid_aanvraag')
sys.path.append(r'C:\Users\gabri\Desktop\amsti_algo\Amsti_Repo\wpi_uitkeringsfraude')

file_path = 'bias_analysis/input/20220523_model_used_in_prepilot.pkl'


# Load the file with joblib
model = load_joblib_with_remapping(file_path)

print(model)