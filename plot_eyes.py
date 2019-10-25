import matplotlib.pyplot as plt
import json

file_name = 'output\\clmtrackr\\5.json'
file = open(file_name)
data = json.load(file)

left_eye = [23, 63, 24, 64, 25, 65, 26, 66, 23]
right_eye = [30, 68, 29, 67, 28, 70, 31, 69, 30]

# plot left eye
xl = []
yl = []

for index in left_eye:
    xl.append(data[index][0])
    yl.append(data[index][1])

plt.plot(xl, yl, label='left_eye')

# plot right eye
xr = []
yr = []

for index in right_eye:
    xr.append(data[index][0])
    yr.append(data[index][1])

plt.plot(xr, yr, label='right_eye')

# pupils
plt.plot([data[27][0], data[32][0]], [data[27][1], data[32][1]], label='left_pupil')

plt.title(file_name)

plt.legend()

plt.show()
