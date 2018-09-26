Code for analysis of the data for the n=2 Lamb Shift measurement is presented here.

2018-09-26 (15:28)

- Fixed a bug with pd.DataFrame.set_index(general_index)
	This is a bad thing to do, because it simply resets the index of the dataframe to the one in general_index, thus loosing indexing accuracy
- Improved speed of execution of certain function
