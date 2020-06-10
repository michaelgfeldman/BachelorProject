import requests
url = 'http://localhost:5000/results'
f = open('essay_example.txt','r')
essay_example = f.readlines()[0]
essay_example = essay_example.lower()
r = requests.post(url,json={'eesay':essay_example})
print(r.json())