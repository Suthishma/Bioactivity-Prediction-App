import pickle

# Open the pickle file for reading in binary mode ('rb')
with open('best_model.pkl', 'rb') as f:
    # Load the contents of the pickle file
    loaded_object = pickle.load(f)

# Now you can use the loaded_object in your code
print(loaded_object)
