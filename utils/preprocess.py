import glob
import pandas as pd
import os
import sklearn.model_selection
def image_folder_to_df(root):
    paths = glob.glob(root + "*/*")
    print(len(paths))
    paths = [p.replace(root, '') for p in paths]
    labels = [os.path.split(p)[0] for p in paths]
    class2index = set(labels)
    class2index = {k: v for v, k in enumerate(class2index)}
    encoded_label = [class2index[l] for l in labels]
    print(paths[:5])
    print(labels[:5])
    print(encoded_label[:5])
    df = pd.DataFrame ([paths, labels, encoded_label]).transpose()
    df.columns = ['path','label','label_idx']
    return df

if __name__ == '__main__':
    root = '/content/images/all/'
    df = image_folder_to_df(root)
    print(df.head())
    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df['label_idx'])
    train, val = sklearn.model_selection.train_test_split(train, test_size=0.2, random_state=42, shuffle=True, stratify=train['label_idx'])
    train.to_csv(os.path.join(root, 'train.csv'), index=False)
    print(train.head())
    test.to_csv(os.path.join(root, 'test.csv'), index=False)
    print(test.head())
    val.to_csv(os.path.join(root, 'val.csv'), index=False)
    print(val.head())