data = open("data.csv", "w")
for i in range(3):
    f = open("image_deltas/" + str(i) + ".txt", "r")

    stuff = f.read().replace(",","").replace("(", "").replace(")","").replace("[","").replace("]","")
    data.write(stuff + "\n")
    f.close()
data.close()
