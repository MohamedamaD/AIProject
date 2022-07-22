from sklearn.preprocessing import LabelEncoder
def scaling(file,col):
    for c in col:
        file[c] = (file[c] - file[c].min()) / (file[c].max() - file[c].min()) # equation of normalization
        
# def remove_zeros(file,col):
#     for c in col:
#         for 
        
def split_data(file,col):
    for c in col:
        file[c] = file[c].str.split('|') 
        
def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))