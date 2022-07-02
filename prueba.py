from re import X


def ivan():
    return 1, *[1,2]
print(ivan())


print(all([None,None]))