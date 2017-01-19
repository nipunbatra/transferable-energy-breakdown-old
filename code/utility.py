import pandas as pd
import matplotlib.pyplot as plt

def plot_df_table(df, title, filename):
	nrows, ncols = len(df)+1, len(df.columns)
	hcell, wcell = 0.3, 1.
	hpad, wpad = 0, 0

	#put the table on a correctly sized figure
	fig=plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
	plt.gca().axis('off')
	matplotlib_tab = pd.tools.plotting.table(plt.gca(),df, loc='center')
	#pp.savefig()
	plt.tight_layout()
	plt.title(title)
	plt.savefig(filename,bbox_inches="tight", dpi=800)