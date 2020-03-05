import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def flatten(listArg):
    temp = []
    for item in listArg:
        for item2 in item:
            temp.append(item2)
    return temp
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pointsX = flatten([[x for y in range(9)] for x in range(3)])
print(pointsX)
pointsY = flatten([[0, 1, 2] for x in range(9)])
pointsZ = flatten([[0, 1, 2] for x in range(9)])

ax.scatter(pointsX, pointsY, pointsZ)
plt.show()