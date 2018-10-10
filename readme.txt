Code for analysis of the data for the n=2 Lamb Shift measurement is presented here.

2018-10-10 (12:27)

- Lots of changes to various codes. Better power calibration. Improved code for common-mode phase shift analysis, etc.


2018-09-28 (15:07)
- Improved the common_mode_phase_shift_analysis.py code. Now the analysis of data will be much faster, because the previously analyzed data is stored in a .csv file.


2018-09-28 (14:07)

- Fixed some bugs. Now the code can be ran, except for pre-910 switching analysis, where it is still using the old averaging function for averaging the FOSOF phases. Need to fix this later.

2018-09-28 (12:38)

- Added better fosof data analysis code version control. I assume that all of the previous code was of v0.1. This code is changed to v0.2


2018-09-27 (00:38)

- Additional speed improvements. Compared to about a week ago, the FOSOF analysis code runs a bit more than 10 times faster. There are still little things to fix here and there, however. Example of this is to run the fosof_data_set_analysis.py for different types of experiment folders to make sure there are not errors. I can also try to modify the saving routine, so that instead of saving text files, I will save pickled objects.

2018-09-26 (15:28)

- Fixed a bug with pd.DataFrame.set_index(general_index)
	This is a bad thing to do, because it simply resets the index of the dataframe to the one in general_index, thus loosing indexing accuracy
- Improved speed of execution of certain function

