"""
\file astronomer.py An example python file for doxgen commenting

The constants in this file are mainly default parameters for astronomer characteristics, like age and height.
"""

##The assumed age of an astromomer in years
#
##This value is assumed for all astronomers unless an alternative value is chosen for them
DEFAULT_ASTRONOMER_AGE = 35

##The defalt height of an astronomer in meters.
DEFAULT_ASTRONOMER_HEIGHT = 1.8

class Astronomer(object):
	"""A single Astronomer.
	
	An astronomer has any number of paper and exactly one name.
	It has a specialism, height, and age, and lifespan which is determined 
	primarily by its height.  Astronomers use Telescope object to observe.
	"""
	def __init__(self, name, papers=None):
		"""
		Construct an astronomer.
		
		Make up an astronomer object by specifying its components. All the normal properties of
		astronomers are assumed.
		
		\param name (String) Mandatory - the astronomer's surname
		\param papers (List of strings) Optional - a list of papers by the astronomer
		\return (Astronomer) A new astronomer instance
		"""
		
		##(String) The astronomer surname
		#
		## The astronomer surname including any hyphenation.
		self.name=name  
		##(Number) The astronomer's age, which has a default value.
		self.age=DEFAULT_ASTRONOMER_AGE  
		
		if self.papers is not None: 
			##(List of strings) The names of papers written by the astronomer
			self.papers=papers[:]
		else:
			self.papers=[]
		
	def write_paper(self,name):
		""" Record that the astronomer has written a new paper.

		Add the paper specified by the name parameter to the Astronomer's list of papers.
		\param name (String) The name of the newly written paper
		"""
		self.papers.append(name)
