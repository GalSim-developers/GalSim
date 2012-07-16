# int and float scalar fields
config.r1 = 53
config.r2 = 3.14159

# r3 is a nested node with a single string field
config.r3.a1 = "huzzah!"

# r4 is a ListField that only allows floats
config.r4 = [1.2, 1.3, 1.4]
config.r4[2] = 1.41  # updates the last element 
config.r4[3] = 1.5   # implicitly appends a new element
config.r4.append(1.6)  # explicitly append

# r5 is a nested node with pluggable options
config.r5 = A  # now r5 is an instance of A; no quotes needed in config file load
config.r5.p1 = -2
config.r5.p2a = 3.2
config.r5 = B  # now r5 is an instance of B; this blows away the above three lines
config.r5.p2b = "bar"

# r6 is a list of pluggables
config.r6 = [A, B, B]  # makes a 3-element list of default-constructed things
config.r6[0].p2a = 17.5
config.r6[1].p2a = 17.5
config.r6[3] = A   # implicitly append; equivalent to '= A()'
config.r6[3].p1 = -4
config.r6[4] = A(p1=3, p2a=3.5)  # implicit append, more concise syntax
