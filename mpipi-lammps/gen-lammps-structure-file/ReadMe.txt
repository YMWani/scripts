README:

seq2config.py is a python file that will read in an amino acid sequence and provide a LAMMPs structure file to
simulate that sequence in Mpipi. Before you begin, you MUST ensure that the following files are in the same 
directory/folder as seq2config.py:

aa_charges_one_letter.txt
aa_charges_three_letters.txt
aa_types_one_letter.txt
aa_types_three_letters.txt

These files provide the program with a 'key' that will enable it to translate an amino acid sequence into 
an Mpipi sequence with the proper types and charges. 

----------REQUIRED PACKAGES----------
python3
numpy

----------PROGRAM INPUTS----------
METHOD 1: a file seq.txt that contains the amino acid sequence to convert to Mpipi
METHOD 2: command line arguments specifying the amino acid code representation (one or three letter codes) and
          the amino acid sequence to convert to Mpipi

----------PROGRAM OUTPUTS----------
A LAMMPs structure file properly listing the amino acids in the sequence provided with bonds, charges,
and positions. Please note that the box size may need to be adjusted to achieve the desired density.
The program will adapt the box size to fit whatever amino acid sequence is provided, but manipulations to
the structure file may be required depending on the application.
The output file will be named myconfig.dat





----------HOW TO RUN THE PROGRAM----------

There are two ways of operating this program. 

----------METHOD 1----------

The first is to provide a sequence text file. This sequence text
file must be in the same directory/folder as seq2config.py. The sequence text file must be named seq.txt . Within
that file, the amino acids must be listed as 

{Amino acid 1}
{Amino acid 2}
...
...
...
{Amino acid N - 1}
{Amino acid N}

There should be no additional spaces and no header/footer to the file. The amino acid may be provided as either
a one letter code or a three letter code. The program will assume that the first amino acid in the file is the 
format for the rest of the amino acids in the file. For example, the amino acid sequence of

ALA
ARG
PHE

would be acceptable. The program will read in the ALA and assume that the rest of the file contains three letter
codes. Similarly, the following would be acceptable:

A
R 
F

as all the amino acids following the first entry continue to be one letter codes. The following, however, would
not be acceptable and will cause errors for the program:

A
ARG
F

----------METHOD 2----------

This method takes in the sequence of the polypeptide you are describing straight from the command line. As such,
two arguments MUST be provided to the program when you run it. The first is whether you are providing a one 
letter code description for the polypeptide you are entering or a three letter code description. Thus, the program
looks for a 1 or a 3 as the first input.

The second argument is the amino acid list you are inputting. This should be provided without spaces.

Valid inputs would therefore look like:

seq2config.py 1 ARF
seq2config.py 3 ALAARGPHE

The program will automatically identify the length of these sequences and delimit the sequence to properly obtain
the correct amino acid sequence.

An invalid example would look like:

seq2config.py 1 AARGF

----------NOTES----------

Please note that using Method 2 overrides method one for this program. Therefore, if you have a file seq.txt
already in the working directory/folder, do not provide any input commands to the program and it will run.

----------TL;DR----------

There are two methods of operation. The first requires a seq.txt file in the working directory. The second is to
specify the amino acid code representation (one or three letter codes) and the amino acid sequence as command
line arguments to the program. The second method will override the first (i.e., provide no command line arguments
if you wish to use method one with a seq.txt file).
