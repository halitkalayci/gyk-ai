batch = [
    (1,'a'),
    (2,'b'),
    (3,'c')
]

# [1,2,3]
# [a,b,c]

x, y = zip(*batch)
print(x)
print(y)