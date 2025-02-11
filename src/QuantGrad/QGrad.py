import math
from graphviz import Digraph

class Value:
    def __init__(self, data, _children = (), _operation = '' , _label = ''):
        self.data = data                   # data to be stored 
        self.grad= 0.0                     # to store gradient
        self._backwards = lambda : None    # Function to calculate gradient
        self._prev = set(_children)         # store the _previous datas (childrex) in a set
        self._operation = _operation       # The operation associated with this instance. Defaults to an empty string.
        self._label = _label


    def __repr__(self):
        # used for printing the object 
        if self._label:
            return f"Value(data={self.data} label={self._label} )"
        return f"Value(data={self.data})"

    #OPERATIONS
    def __add__(self, other , _label =''):
        # a+b internally a.__add__(b)
        
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data + other.data , _children = (self, other) , _operation = '+', _label =_label)

        def _backwards():
            self.grad  += out.grad
            other.grad += out.grad # why += if we use the same vaiable more than once it will not accumulate the gradient if we just use =  
        out._backwards = _backwards#soo we have to accumulate them 
        return out  
    
    def __radd__(self, other):
        return self if other == 0 else self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self,other):
        return self+(-other)
      

    def __mul__(self, other, _label = ''):
        # a*b internally a.__mul__(b)
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data , _children = (self, other) , _operation = '*' , _label =_label)
        def _backwards():
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        out._backwards = _backwards
        return out
    
    def __rmul__(self,other): # Fallback for __mul__  #self == other , other == self
        return self*other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        data = self.data**other
        out = Value(data=data, _children=(self,), _operation="**")
        def _backwards():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backwards = _backwards
        return out

    
    def __truediv__(self, other):
        return self*(other**-1)
    
    def exp(self):
        x=self.data
        out = Value(data=math.exp(x), _children=(self,), _operation="exp")
        def _backwards():
            self.grad += (out.data) * out.grad
        out._backwards = _backwards
        return out
    #ACTIVATION FUNCTIONS
    def tanh(self):
        x=self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t , _children=(self,), _operation= "tanh")

        def _backwards():
            self.grad += (1-t**2) * out.grad
        out._backwards = _backwards
        return out
    
    #OVERALL BACKWARD FUNCTION
    def backwards(self):
        """
        Performs backpropagation to compute the gradients of all nodes in the computational graph.
        """
        visited =set()
        topo_ordering = []
        #Topological sortiing 
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                
                topo_ordering.append(node)
        self.grad =1.0
        build_topo(self)
        # print(list(reversed(topo_ordering)))  # Commented out to avoid printing during backpropagation
        for node in reversed(topo_ordering):
            node._backwards()
    
    

def trace(root):
    """
    Traverses a tree structure starting from the given root node and builds sets of nodes and edges.

    Args:
        root: The root node of the tree to be traversed.

    Returns:
        A tuple containing two sets:
            - nodes: A set of all nodes in the tree.
            - edges: A set of tuples representing the edges between nodes in the tree.
    """
    nodes , edges = set() ,set() # to 
    def build(node):
        if node not in nodes:
            nodes.add(node)
            for child in node._prev:
                edges.add((child ,node))
                build(child)
    build(root)
    
    return nodes, edges

def draw_dot(root , format = 'svg', rankdir = 'LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR','TB']
    
    nodes, edges= trace(root)
    dot = Digraph(format = format, graph_attr= {'rankdir':rankdir})

    
        
    for node in nodes: 
        uid_for_node= str(id(node))
        dot.node(name = uid_for_node, label = f"{{ {node._label} | {node.data:.4f} | {node.grad:.4f} }}", shape = "record" )

        if node._operation:
            uid_for_operation = uid_for_node + node._operation
            dot.node(name= uid_for_operation, label = f"{node._operation}")
            dot.edge(uid_for_operation, uid_for_node) # connect operation to node
    for node1 , node2 in edges:
        dot.edge(str(id(node1)), str(id(node2))+node2._operation)
        
    
    return dot
        
    