a=[[12,2,15,4,5],[23,2,9,6]]
for b in a :
    counter=0
    for c in b :
        if c < 5 :
            b[counter]=0
        counter+=1
print(a)