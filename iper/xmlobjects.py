#!/usr/bin/python

from copy import deepcopy
from lxml import etree

def fromXml(xml_string):
    return etree.fromstring(xml_string)

def fromXmlFile(fname):
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(fname, parser)
    return tree
  
def toStr(el, prettyPr=True):
    _s = etree.tostring(el, pretty_print=prettyPr)
    if type(_s) == bytes:
      _s = _s.decode()
    return _s

class XMLNode(object): 

    def __init__(self,name):
        self.el = etree.Element(name)
        
    def add(self, name):
        node = XMLNode(name)
        self.el.append(node.el)    
        return node

class XMLObject(object):
    """ An xml based object used to modelize data.
        Can hold properties and sub elements
        
        
        >>> m = XMLObject('lungsmodel')
        >>> lungs = m.add('lungs')
        >>> airways = m.add('airways')
        >>> rlung = lungs.add('rigth')
        >>> llung = lungs.add('left')
        >>> trachea = airways.add('trachea')
        
        >>> lungs.el.set('id','1')
        
        >>> print(str(m))
        <lungsmodel>
          <lungs id="1">
            <rigth/>
            <left/>
          </lungs>
          <airways>
            <trachea/>
          </airways>
        </lungsmodel>
        <BLANKLINE>
        
        """ 
    
    def __init__(self, name='NewObject'):
        self.rootNode = XMLNode(name)
        self.tree = self.rootNode.el.getroottree()
                
    def add(self,name):
        return self.rootNode.add(name)
        
    def __str__(self):
        return toStr(self.rootNode.el, prettyPr=True)
        
    def merge(self, other):
      otree = other.getroottree()
      rpath = otree.getpath(other)
      for elem in other.findall(".//"):
        path = otree.getpath(elem).replace(rpath,"./")
        _els = self.tree.findall(path)
        if not _els: 
          self.rootNode.el.append(deepcopy(elem))
        #for _el in :
        #  print _el.attrib
        
      
    def createObjectFromXml(self,xmlstring,name='newobj'):
        obj = self.createObject(name)
        obj.addElement(etree.fromstring(xmlstring))
        return obj

    @staticmethod
    def createObjectFromXmlFile(fname):
        tree = fromXmlFile(fname)
        el = tree.getroot()
        obj = XMLObject("NewObject")
        obj.rootNode.el = el
        return obj
    
    def setAttributes(self,el,vals):
        for k,v in vals.items():
            el.setAttribute(k,v)
        
    def addElement(self, el1, name):
        newEl = self.doc.createElement(name)
        try:
            r = el1.appendChild(newEl)
        except:
            el1Doc = el1._get_ownerDocument()
            r = el1.appendChild(el1Doc.importNode(newEl,True))
        
        return r

    def toXmlFile(self, filename):
        with open(filename,'w') as fp:
          fp.write(str(self))
          

def _test():
    import doctest
    return doctest.testmod()

if __name__ == '__main__':
    _test()
