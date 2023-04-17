import os

with open('everyday_val_Bottle.txt', 'r') as file:
    lines = file.readlines()

with open('everyday_val_Bottle_juhong.txt', 'w') as file:
    ll = []
    for line in lines:
        #print()
        ll.append('/'.join(line.split()[1].split('/')[:-1])+'\n')
    print()
    for i in sorted(list(set(ll))):
        file.write(i)