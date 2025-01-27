# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:11:56 2019

@author: hemaxi
"""

from lxml import etree



def beginXML(img, width, height):
    annotation = etree.Element('annotation')

    fo = etree.Element('folder')
    fo.text =  '/test/training/images'

    annotation.append(fo)

    f = etree.Element('filename')
    f.text = str(img)

    annotation.append(f)

    size = etree.Element('size')
    w = etree.Element('width')
    w.text = str(width)
    h = etree.Element('height')
    h.text = str(height)
    d = etree.Element('depth')
    d.text = str(3)


    size.append(w)
    size.append(h)
    size.append(d)

    annotation.append(size)
    
    return annotation



def createXML(xmin, xmax, ymin, ymax, img, width, height, annotation):
    

    #seg = etree.Element('segmented')
    #seg.text = str(0)

    #annotation.append(seg)

    object = etree.Element('object')
    n = etree.Element('name')
    p = etree.Element('pose')
    t = etree.Element('truncated')
    d_1 = etree.Element('difficult')
    bb = etree.Element('bndbox')

    n.text = 'Nuclei'
    p.text = 'center'
    t.text = str(1)
    d_1.text = str(0)

    xmi = etree.Element('xmin')
    ymi = etree.Element('ymin')
    xma = etree.Element('xmax')
    yma = etree.Element('ymax')

    xmi.text = str(xmin)
    yma.text = str(ymax)
    ymi.text = str(ymin)
    xma.text = str(xmax)

    bb.append(xmi)
    bb.append(ymi)
    bb.append(xma)
    bb.append(yma)

    object.append(n)
    object.append(p)
    object.append(t)
    object.append(d_1)
    object.append(bb)

    annotation.append(object)

    return annotation