import pandas as pd

def checkitout():
    global a
#    a=[]
    a.append(100)
    global df1
    df1 = df1.append(df1)
    print "1................."
    print df1
    print a
    anothercheck()

def anothercheck():
    global a
    a.append(5)
    global df1
    df1 = df1.append(df1)
    print "2......................"
    print df1
    print a



#global a
a=[]
#a=[5]
print a
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],
'D': ['D0', 'D1', 'D2', 'D3']})
print "main"
print df1

checkitout()

checkitout()

print a