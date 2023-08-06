import os
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


log_filename = "trained_log.csv"


# ファイル監視イベントハンドラ
class EventHandler(FileSystemEventHandler):
	def on_any_event(self,e):
		global	log_filename
		if	os.path.basename(e.src_path) == log_filename:
			event.set()

event = threading.Event()
handler = EventHandler()
observer = Observer()
observer.schedule( handler, path=os.path.abspath("./") )
observer.start()


# CSV をグラフ表示
def PlotLogGraph( ax, x, tl, vl, tlabel, vlabel ):
	yscale_log = False
	if	len(tl) >= 100:
		tlavg = np.mean( tl[len(tl)-50:] )
		lstd = max( np.std( tl[len(tl)-50:] ),
					np.std( vl[len(tl)-50:] ) )
		yscale_log = (tlavg + lstd * 5.0 < max(tl))

	if	yscale_log:
		ax.set_yscale( "log" )
	else:
		tvmin = min( min(tl), min(vl) )
		tvmax = max( max(tl), max(vl) )
		ax.set_ylim( min( tvmin - abs(tvmin) * 0.1, 0.0 ),
					 tvmax + abs(tvmax) * 0.2 )

	ax.plot( x, vl, color="red", label=vlabel )
	ax.plot( x, tl, color="blue", label=tlabel )

	ax.set_xlabel( "epoch" )
	ax.set_ylabel( "evaluation" )
	ax.legend(loc = 'upper right') 
	ax.grid()


def PlotLogCSV():
	global	log_filename
	global	fig_created
	global	fig, ax, ax2

	try:
		data = pd.read_csv( log_filename )

	except:
		return

	if	(data.shape[0] < 1) or (data.shape[1] < 2):
		return

	tl = data.iloc[:,0]
	vl = data.iloc[:,1]

	if	fig_created == 0:
		if	(data.shape[1] < 4) or pd.isnull( data.iloc[:,2][0] ):
			fig, ax = plt.subplots()
			fig_created = 1

		else:
			fig = plt.figure( figsize = (8,6) )
			ax = fig.add_subplot(2, 1, 1)
			ax2 = fig.add_subplot(2, 1, 2)
			fig_created = 2

		fig.canvas.mpl_connect( "close_event", handle_close )


	if	len(tl) >= 2:
		x = list(range(len(tl)))

		ax.cla()
		PlotLogGraph( ax, x, tl, vl, data.columns[0], data.columns[1] ) ;

		if	(fig_created == 2) and (not pd.isnull( data.iloc[:,2][0] )):
			ax2.cla()
			PlotLogGraph( ax2, x,
						data.iloc[:,2], data.iloc[:,3],
						data.columns[2], data.columns[3] )

		plt.draw()
		plt.pause( 0.3 )


# ウィンドウを閉じた時のハンドラ
def handle_close(evt):
	global is_closed
	is_closed = True

fig_created = 0
is_closed = False

PlotLogCSV()


# ループ
while not is_closed:
	while not event.wait(0.1):
		plt.pause( 0.5 )
		if is_closed:
			break
	if is_closed:
		break

	event.clear()

	PlotLogCSV()

