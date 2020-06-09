from sklearn.datasets import load_files
d = load_files('aclImdb-small/test', categories=['neg','pos'], shuffle=True, encoding='utf-8')
print(d['filenames'][:3])
