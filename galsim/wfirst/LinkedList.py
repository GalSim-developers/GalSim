class Node:
    def __init__(self):
        self.data = None # the data content
        self.next = None # the reference to the next node

class LinkedList:
    def __init__(self, data=None):
        self.cur_node = None
        if isinstance(data, LinkedList):
            self.cur_node = data.cur_node
        elif isinstance(data, list):
            for i in data[::-1]:
                self.add_node(i)
        elif data is not None:
            raise TypeError("linked_list constructor accepts only None, linked_list or list")

    def addNode(self, data):
        new_node = Node() # create a new node
        new_node.data = data
        new_node.next = self.cur_node # link the new node to the 'previous' node.
        self.cur_node = new_node #  set the current node to the new one.

    def __str__(self):
        node = self.cur_node
        s = ""
        while node:
            s += str(node.data)+'\n'
            node = node.next
        return s[:-1]

    def listPrint(self):
        node = self.cur_node
        while node:
            print node.data
            node = node.next

    def listLen(self):
        count = 0 
        node = self.cur_node
        while node:
            count += 1 
            node = node.next
        return count
        
    def deleteNode(self,location):
        if location == 0:
            try:
                self.cur_node = cur_node.next
            except AttributeError:
                # The list is empty
                self.cur_node = None
            finally:
                return 

        node = self.cur_node        
        try:
            for _ in xrange(location-1):
                node = node.next
        except AttributeError:
            # The list isn't long enough
            raise ValueError("List does not have index {0}".format(location))

        try:
            node.next = node.next.next
        except AttributeError:
            # The desired node is the last one.
            node.next = None
