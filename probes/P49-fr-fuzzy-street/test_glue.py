import re
def fix(x):
    return re.sub(r"\b([ld])['’\s]+", r"\1", x)

print(fix("de l'argonne"))
print(fix("de l argonne"))
print(fix("de l' argonne"))
print(fix("de d'agen"))
print(fix("d'agen"))
