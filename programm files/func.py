from sklearn.preprocessing import LabelEncoder
def scaling(file,col):
    for c in col:
        file[c] = (file[c] - file[c].min()) / (file[c].max() - file[c].min()) # equation of normalization
        
def split_data(file,col):
    for c in col:
        file[c] = file[c].str.split('|') 
        