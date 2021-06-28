import pickle

sample_name = []
sample_label = []
sample_name.append("test.skeleton")
sample_label.append(51)  # TODO: pipeline to input labels from API

with open(r'C:\Users\chuaz\file-upload\dgnn\data\test_label.pkl', 'wb') as f:
    data = pickle.dump((sample_name, list(sample_label)), f)

with open(r'C:\Users\chuaz\file-upload\dgnn\data\test_label.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
