
def sliding_window(img, size, step):
	xall = np.expand_dims(img, axis=0)

	for y in range(0, 280, step):
	     for x in range(0, 280, step):
	         x1n = np.copy(img)
	         x1n = np.expand_dims(x1n, axis=0)
	         x1n[:,y:y + size, x:x + size]=0
	         xall= np.concatenate((xall, x1n), axis=0)

	return xall[1:len(xall)]


pic1_window = sliding_window(pic1, 60, 10)
pic1_pred = np.round(model.predict(pic1_window))
pic1_re = np.reshape(pic1_pred, (28,28))
pic1_re_flip = np.fliplr(pic1_re)


pic2_window = sliding_window(pic2, 60, 10)
pic2_pred = np.round(model.predict(pic2_window))
pic2_re = np.reshape(pic2_pred, (28,28))
pic2_re_flip = np.fliplr(pic2_re)



fig, ax = plt.subplots()
cax = ax.imshow(pic3_re, clim=(4.0, 7), cmap='hot')
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ticks=[7,6,5,4,3]
cbar = fig.colorbar(cax, ticks=ticks, boundaries=[4,5,6,7,8])
loc = [7.5, 6.5, 5.5, 4.5]
cbar.set_ticks(loc)
cbar.ax.set_yticklabels(['GT', '-1', '-2', '-3'])
plt.show()


# fig, ax = plt.subplots()
# cax = ax.imshow(pic1_re, clim=(11.0, 17), cmap='hot')
# ax.axes.get_xaxis().set_ticks([])
# ax.axes.get_yaxis().set_ticks([])
# ticks=[18,17,16,15,14,13,12]
# cbar = fig.colorbar(cax, ticks=ticks, boundaries=[11,12,13,14,15,16,17,18])
# loc = [17.5, 16.5, 15.5 , 14.5, 13.5, 12.5, 11.5]
# cbar.set_ticks(loc)
# cbar.ax.set_yticklabels(['GT', '-1', '-2', '-3','-4','-5','-6'])
# plt.show()

fig, ax = plt.subplots()
cax = ax.imshow(pic2_re_flip, clim=(8.0, 12), cmap='hot')
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ticks=[13,12,11,10,9,8]
cbar = fig.colorbar(cax, ticks=ticks, boundaries=[8,9,10,11,12,13])
loc = [12.5, 11.5, 10.5, 9.5, 8.5]
cbar.set_ticks(loc)
cbar.ax.set_yticklabels(['GT', '-1', '-2', '-3','-4','-5'])
plt.show()

