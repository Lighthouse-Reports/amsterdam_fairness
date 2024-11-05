import joblib
from sklearn.pipeline import Pipeline
import sys
import pickle

#Change path as needed
sys.path.append(r'C:\Users\ta0sh\Desktop\Lighthouse\Amsterdam\Gabes_setup\Amsti_Repo\wpi_onderzoekswaardigheid_aanvraag')

# Custom unpickler to remap module names
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(f"Trying to load {module}.{name}")
        if module == 'wpi_uitkeringsfraude.preprocessing.custom_transformers':
            module = 'wpi_onderzoekswaardigheid_aanvraag.preprocessing.custom_transformers'
        print(f"Remapped to {module}.{name}")
        return super().find_class(module, name)

def load_new(file):
    try:
        with open(file, 'rb') as f:
            return CustomUnpickler(f).load()
    except ModuleNotFoundError as e:
        print(f"ModuleNotFoundError: {e}")

# Attempt to load the model using joblib
try:
    model = joblib.load('20220531_model_after_reweighing.pkl')
    print("File loaded using joblib.")
except Exception as e:
    print(f"Joblib load error: {e}")
    model = load_new('20220531_model_after_reweighing.pkl')

# Examining the model
if model:
    if isinstance(model, dict):
        print("The unpickled object is a dictionary.")
        if 'model' in model:
            pipeline = model['model']
            if isinstance(pipeline, Pipeline):
                print("The dictionary contains an sklearn pipeline.")
            else:
                print("The dictionary does not contain an sklearn pipeline.")
        else:
            print("The dictionary does not contain a pipeline key.")
    else:
        print("The unpickled object is not a dictionary.")
else:
    print("Failed to unpickle the object.")
