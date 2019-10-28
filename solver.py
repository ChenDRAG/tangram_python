import copy
import math
from math import sqrt

import cv2
import numpy as np

# qiuqiu! pangqiuqiu :)

class Picture:
    #pic numpy array container
    #also have methods to detect edges and nodes(including hidden ones) in image
    def __init__(self, pic):
        self.pic_ori = pic
        self.edges1 = set([])
        self.edges2 = set([])
        self.nodes = set([])
        self.unit_lenth = self.cal_unit_l()
        self.raw_keypoints = []
        self.goodpoints = np.array([])

    def analysis_pic(self):
        #assume the theta is default to 0 for now 
        #don't consider several graph in one pic for now
      
        #Find keypoints and exclude those fake ones and include unfound
        self.raw_keypoints = np.int0(cv2.goodFeaturesToTrack(self.pic_ori, 23, 0.17, self.unit_lenth // 4)) 
        self.raw_keypoints = [p.ravel() for p in self.raw_keypoints]
        self.valid_keypoints = Picture.find_valid(self.raw_keypoints, self.unit_lenth)
        
        for theta, points in self.valid_keypoints:
            self.find_nodes(points, theta)
        
        return self.nodes, self.edges1, self.edges2

    def cal_unit_l(self):
        #cal unit lenth
        h, w = self.pic_ori.shape
        black = 0
        for x in range(w):
            for y in range(h):
                if self.pic_ori[y][x] == 0:
                    black += 1
        return sqrt(black/8)     

    @staticmethod
    def find_valid(points, unit):
        unit = int(unit)
        assert type(points) == list
        points_remain = points
        goods = [] #list containing list

        def near(p1, p2, thata):
            p = p1 - p2
            p = Picture.regularize(p, unit, theta)
            return np.linalg.norm(p) < unit/10.

        def a_set(points_remain, thata):
            assert type(points_remain) == list
            good = []
            for standard in points_remain: 
                good = [p for p in points if near(p, standard, thata)]
                if len(good) >= 3:
                    break
                else:
                    good = []
            return good

        thetas = [0, 45]
        for theta in thetas:
            while True:
                if len(points_remain) <= 2:
                    return goods
                good = a_set(points_remain, theta)
                if len(good) > 0:
                    points_remaint = []
                    for p in points_remain:
                        lt = [(p==g).all() for g in good]
                        if len(lt) == 0:
                            points_remaint.append(p)
                    points_remain = points_remaint
                    goods.append((theta, good))
                else:
                    break
        return goods
    
    @staticmethod
    def regularize(p, unit, theta = 0):
        ct = math.cos(theta / 180.*3.1415926)
        st = math.sin(theta / 180.*3.1415926)
        R = np.array([[ct, -st], [st, ct]])
        p = p.copy()
        p = R.T.dot(p[::-1])[::-1]
        if abs(p[0] - (p[0]//unit) * unit) < abs(p[0] - (p[0]//unit + 1) * unit):
            p[0] = p[0] - (p[0]//unit) * unit
        else:
            p[0] = p[0] - (p[0]//unit + 1) * unit

        if abs(p[1] - (p[1]//unit) * unit) < abs(p[1] - (p[1]//unit + 1) * unit):
            p[1] = p[1] - (p[1]//unit) * unit
        else:
            p[1] = p[1] - (p[1]//unit + 1) * unit

        p = R.dot(p[::-1])[::-1]
        return p  
    
    def find_nodes(self, points, theta):
        ct = math.cos(theta / 180.*3.1415926)
        st = math.sin(theta / 180.*3.1415926)
        R = np.array([[ct, -st], [st, ct]])

        re_p = np.array([Picture.regularize(p, self.unit_lenth, theta) for p in points])
        ori_p = np.mean(re_p, axis = 0).astype(np.int)
        ori_p = R.T.dot(ori_p[::-1])
  
        ori_p[0] = ori_p[0] - int(self.unit_lenth) * 7
        ori_p[1] = ori_p[1] - int(self.unit_lenth) * 7

        ori_p = R.dot(ori_p)[::-1]
        # if(ori_p[0] < 0):
        #     ori_p[0] = ori_p[0] + int(self.unit_lenth)
        # if(ori_p[1] < 0):
        #     ori_p[1] = ori_p[1] + int(self.unit_lenth) 

        #assert ori_p[0] >= 0
        #assert ori_p[1] >= 0        
        
        # h,w = self.pic_ori.shape
        # num_h = h // self.unit_lenth + 1 
        # num_w = w // self.unit_lenth + 1
        # num_h = int(num_h)
        # num_w = int(num_w)
        # if (num_h - 1) * int(self.unit_lenth) + ori_p[1] >= h:
        #     num_h = num_h - 1
        # if (num_w - 1) * int(self.unit_lenth) + ori_p[0] >= w:
        #     num_w = num_w - 1    
        # don't wanna write compliated algorithm, thus simplify 
        h,w = self.pic_ori.shape
        num_h = h // self.unit_lenth + 14
        num_w = w // self.unit_lenth + 14
        num_h = int(num_h)
        num_w = int(num_w)            
        self.p_candidates = np.zeros((num_h ,num_w)).astype(np.bool)
        for i in range(num_h):
            for j in range(num_w):
                if self.point_valid(self.get_coord(i, j, ori_p, theta)):
                    self.p_candidates[i][j] = True
        self.pack_up_nodes_and_edges(self.p_candidates, ori_p, theta) #pack up nodes and edges base on candidate and ori_p
        #self.p_candidates = self.p_candidates.tolist()
        return self.p_candidates

    def get_coord(self, i, j, ori_p, theta):
        ct = math.cos(theta / 180.*3.1415927)
        st = math.sin(theta / 180.*3.1415927)
        R = np.array([[ct, -st], [st, ct]])
        return (ori_p + R.dot(np.array([i, j]) * np.array([int(self.unit_lenth), int(self.unit_lenth)]))[::-1]).astype(np.int)

    def pack_up_nodes_and_edges(self, p_candidates, ori_p, theta):
        #node
        hn = p_candidates.shape[0]
        wn = p_candidates.shape[1]
        nodesM = p_candidates.tolist()
        for h in range(hn):
            for w in range(wn):
                if p_candidates[h][w]:
                    nodesM[h][w] = Node(self.get_coord(h, w, ori_p, theta))
                    self.nodes.add(nodesM[h][w])
                
        #edge1 and neibour   
        for h in range(hn - 1):
            for w in range(wn):
                if p_candidates[h][w]:
                    if p_candidates[h + 1][w] and self.point_valid((self.get_coord(h, w, ori_p, theta)+self.get_coord(h + 1, w, ori_p, theta))//2): 
                        nodesM[h][w].add_neighbours(nodesM[h + 1][w])
                        nodesM[h + 1][w].add_neighbours(nodesM[h][w])
                        self.edges1.add((nodesM[h][w], nodesM[h+1][w]))
        for h in range(hn):
            for w in range(wn - 1):
                if p_candidates[h][w]:
                    if p_candidates[h][w + 1] and self.point_valid((self.get_coord(h, w, ori_p, theta)+self.get_coord(h, w + 1, ori_p, theta))//2): 
                        nodesM[h][w].add_neighbours(nodesM[h][w + 1])
                        nodesM[h][w + 1].add_neighbours(nodesM[h][w])
                        self.edges1.add((nodesM[h][w], nodesM[h][w + 1]))
                        
        #edge2
        for h in range(hn -1):
            for w in range(wn - 1):
                if p_candidates[h][w]:
                    if p_candidates[h + 1][w + 1] and self.point_valid((self.get_coord(h, w, ori_p, theta)+self.get_coord(h + 1, w + 1, ori_p, theta))//2): 
                        self.edges2.add((nodesM[h][w], nodesM[h + 1][w + 1]))    
        for h in range(1, hn):
            for w in range(wn - 1):
                if p_candidates[h][w]:
                    if p_candidates[h - 1][w + 1] and self.point_valid((self.get_coord(h, w, ori_p, theta)+self.get_coord(h - 1, w + 1, ori_p, theta))//2): 
                        self.edges2.add((nodesM[h][w], nodesM[h - 1][w + 1]))     

    def point_valid(self, p):
        if p[1] < 0 or p[1] >= self.pic_ori.shape[0]:
            return False
        if p[0] < 0 or p[0] >= self.pic_ori.shape[1]:
            return False            
        p = p.astype(np.int)
        d = int(self.unit_lenth/8)
        sample = self.pic_ori[max(p[1] - d, 0):min(p[1] + d, self.pic_ori.shape[0]), 
                                max(p[0] - d, 0):min(p[0] + d, self.pic_ori.shape[1])] > 128
        if(np.sum(sample)/sample.size < 0.96):
            return True
        else:
            return False  

    def veriT(self, n1, n2, n3):
        return self.point_valid((n1.coord + n2.coord + n3.coord)//3)

    @staticmethod
    def visualise(pic, points = None, edges = None, filename = None, color_set = None):
        assert type(color_set) == tuple or color_set is None
        color = pic.copy()
        if(len(color.shape) == 2):
            color = np.expand_dims(color, axis = 2)
            color = np.concatenate((color, color, color), axis = 2)
        if points is not None:
            for point in points:
                if len(point.shape) == 2:
                    point = point.ravel()
                x, y = point
                cv2.circle(color, (x, y), 5, color_set if color_set  is not None else (255, 0, 0), -1)
        if edges is not None:
            for edge in edges:
                cv2.line(color, tuple(edge[0].coord.tolist()), tuple(edge[1].coord.tolist()), color_set if color_set is not None else (0, 255, 0), 2)
        if filename is not None:
            cv2.imwrite(filename, color)
        return color

class Node:
    def __init__(self, coord):
        self.coord = coord
        self.neighbours = set([])
    def add_neighbours(self, n):
        self.neighbours.add(n)

#define abstract parent class for blocks 
class BaseShape:
    name = None
    def __init__(self):
        self.children = {}
    def check_containing(self, sstlist):
        assert type(sstlist) == list
        for namet, child in self.children.items():
            if namet == 'SST' and set(sstlist).intersection(set(child)) != set([]):
                return True
            elif namet != 'SST':
                for one in child:
                    if one.check_containing(sstlist) == True:
                        return True
        return False
    def basics(self):
        sstlist = set([])
        for name, child in self.children.items():
            if name == 'SST':
                sstlist = sstlist.union(set(child))
            elif name != 'SST':  
                for one in child:
                    sstlist = sstlist.union(set(one.basics()))
        return list(sstlist)

class ST(BaseShape):
    name = 'ST'
    color = (100,100,0)
    def __init__(self, node1, node2, node3):
        self.nodes = [node1, node2, node3]
        self.children = self.setup_ssts()
        
        #{'SST':[]}
    def setup_ssts(self):
        children = []
        #vetor determined "left" > 0
        if np.cross(self.nodes[0].coord - self.nodes[2].coord, self.nodes[0].coord - self.nodes[1].coord) >0:
            children.append((self.nodes[0], self.nodes[2]))
        else:
            children.append((self.nodes[2], self.nodes[0]))
        if np.cross(self.nodes[1].coord - self.nodes[2].coord, self.nodes[1].coord - self.nodes[0].coord) >0:
            children.append((self.nodes[1], self.nodes[2]))
        else:
            children.append((self.nodes[2], self.nodes[1]))        
        return {'SST':children}
    def ordernodes(self):
        return list(self.nodes)
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for node in other.nodes:
            if node not in self.nodes:
                return False
        return True 

class LT(BaseShape):
    name = 'LT'
    color = (0,100,100)
    def __init__(self, mt1, mt2):
        self.children = {'MT': [mt1, mt2]}
        nodes_ = [i for i in mt1.nodes[:2] if i in mt2.nodes[:2]][0]
        self.nodes = []
        self.nodes.extend([i for i in mt1.nodes[:2] if i!= nodes_])
        self.nodes.extend([i for i in mt2.nodes[:2] if i!= nodes_])
        self.nodes.append(nodes_)
        assert len(self.nodes) == 3
    def ordernodes(self):
        return list(self.nodes)
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        return set(other.nodes) == set(self.nodes)

class MT(BaseShape):
    color = (100,0,100)
    name = 'MT'
    def __init__(self, st1, st2):
        self.children = {'ST': [st1, st2]}
        nodes_ = [i for i in st1.nodes[:2] if i in st2.nodes[:2]][0]
        self.nodes = []
        self.nodes.extend([i for i in st1.nodes[:2] if i!= nodes_])
        self.nodes.extend([i for i in st2.nodes[:2] if i!= nodes_])
        self.nodes.append(nodes_)
        assert len(self.nodes) == 3
    def ordernodes(self):
        return list(self.nodes)
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        return set(other.nodes) == set(self.nodes)

class SQ(BaseShape):
    name = 'SQ'
    color = (150,50,100)
    def __init__(self, st1, st2):
        self.children = {'ST': [st1, st2]}
        self.nodes = set(st1.nodes).union(set(st2.nodes))

    def ordernodes(self):
        return [self.children['ST'][0].nodes[2], self.children['ST'][0].nodes[1], 
                self.children['ST'][1].nodes[2], self.children['ST'][0].nodes[0]]

    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        return other.nodes == self.nodes

class PA(BaseShape):
    name = 'PA'
    color = (100,150,50)
    def __init__(self, st1, st2):
        self.children = {'ST': [st1, st2]}
        nodes_ = set(st1.nodes).union(set(st2.nodes))
        #fist 2 135d last 2 45d
        self.nodes = [n for n in nodes_ if n == st1.nodes[2] or n == st2.nodes[2]]
        self.nodes.extend([n for n in nodes_ if n not in self.nodes])
    def ordernodes(self):
        return [self.children['ST'][0].nodes[2], 
        self.children['ST'][0].nodes[1] if self.children['ST'][0].nodes[1] != self.children['ST'][1].nodes[2] else self.children['ST'][0].nodes[0],
        self.children['ST'][1].nodes[2], 
        self.children['ST'][1].nodes[1] if self.children['ST'][1].nodes[1] != self.children['ST'][0].nodes[2] else self.children['ST'][1].nodes[0],
        ]
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        return set(other.nodes) == set(self.nodes)

# class containing methods to analysis pics and nodes and create the searching space
class Graph:
    def __init__(self, nodes, edges1, edges2):
        assert type(nodes) == set
        assert type(edges1) == set
        assert type(edges2) == set
        
        self.nodes = nodes
        self.edges1 = edges1# set containing set
        self.edges2 = edges2
        self.ssts = []#directed edge
        self.sts = []#
        self.mts = []
        self.lts = []
        self.sqs = []
        self.pas = []
    def setup(self, pic):
        for node1, node2 in self.edges2:
            flag = True
            pointy = [i for i in node1.neighbours if i in node2.neighbours]
            for node3 in pointy:
                if not pic.veriT(node1, node2, node3):
                    flag = False
                    continue
                st = ST(node1, node2, node3)
                if st in self.sts:
                    continue
                self.sts.append(st)
                self.ssts.extend(s for s in st.children['SST'] if s not in self.ssts)
            
            
            if len(pointy) == 2 and flag:
                t1 = [i for i in self.sts if set(i.nodes) == set([pointy[0], node1, node2])]
                t2 = [i for i in self.sts if set(i.nodes) == set([pointy[1], node1, node2])] 
                sq = SQ(t2[0], t1[0])
                if sq in self.sqs:
                    continue
                else:
                    self.sqs.append(sq)

        pas_t = [PA(st1, st2) for st1 in self.sts for st2 in self.sts 
                    if st1 != st2 and (
                    (st1.nodes[2] == st2.nodes[0] and st2.nodes[2] == st1.nodes[0] and st1.nodes[1] not in st2.nodes[1].neighbours)
                    or
                    (st1.nodes[2] == st2.nodes[1] and st2.nodes[2] == st1.nodes[1] and st1.nodes[0] not in st2.nodes[0].neighbours)
                    or
                    (st1.nodes[2] == st2.nodes[0] and st2.nodes[2] == st1.nodes[1] and st1.nodes[0] not in st2.nodes[1].neighbours)
                    or
                    (st1.nodes[2] == st2.nodes[1] and st2.nodes[2] == st1.nodes[0] and st1.nodes[1] not in st2.nodes[0].neighbours)
                    )]
        for pa in pas_t:
            if pa not in self.pas:
                self.pas.append(pa)
            
        mts_t = [MT(st1, st2) for st1 in self.sts for st2 in self.sts
                    if st1 != st2 and (st1.nodes[0] in st2.nodes[:2] or st1.nodes[1] in st2.nodes[:2])
                    and st1.nodes[2] == st2.nodes[2]]

        for mt in mts_t:
            if mt not in self.mts:
                self.mts.append(mt)

        lts_t = [LT(mt1, mt2) for mt1 in self.mts for mt2 in self.mts
                    if mt1 != mt2 and 
                    (mt1.nodes[0] == mt2.nodes[1] or mt1.nodes[0] == mt2.nodes[0]
                    or mt1.nodes[1] == mt2.nodes[1] or mt1.nodes[1] == mt2.nodes[0])
                    and mt1.nodes[2] == mt2.nodes[2]]
        for lt in lts_t:
            if lt not in self.lts:
                self.lts.append(lt)           
        

    #def setup_ssts():

# define one possible status when solving the tangram        
class Status:
    g = None
    q = None
    shape_num = None
    pic = None
    #domains = {'LT':large_triangles, 'MT':medium_triangles, 'SQ':squares, 'PA':parallelograms, 'ST':small_triangles}
    shape_needed_num = {'LT':2, 'MT':1, 'SQ':1, 'PA':1, 'ST':2}
    
    def __init__(self):
        assert type(self.g) == Graph
        self.sst_remain = self.g.ssts #directed edge
        self.st_remain = self.g.sts  #
        self.mt_remain = self.g.mts
        self.lt_remain = self.g.lts
        self.sq_remain = self.g.sqs
        self.pa_remain = self.g.pas
        self.shape_remain = {'LT':self.lt_remain, 'MT':self.mt_remain, 'SQ':self.sq_remain, 'PA':self.pa_remain, 'ST':self.st_remain}
        self.where = {'LT':[], 'MT':[], 'SQ':[], 'PA':[], 'ST':[]}
        self.next_choice = 0
        self.updated = True
        self.newly_filled = None

    def __deepcopy__(self, memo):
        other = Status()
        other.sst_remain = self.sst_remain.copy()
        other.st_remain = self.st_remain.copy()
        other.mt_remain = self.mt_remain.copy()
        other.lt_remain = self.lt_remain.copy()
        other.sq_remain = self.sq_remain.copy()
        other.pa_remain = self.pa_remain.copy()
        other.shape_remain = {'LT':other.lt_remain, 'MT':other.mt_remain, 'SQ':other.sq_remain, 'PA':other.pa_remain, 'ST':other.st_remain}
        other.where = {'LT':self.where['LT'].copy(), 'MT':self.where['MT'].copy(), 'SQ':self.where['SQ'].copy(), 'PA':self.where['PA'].copy(), 'ST':self.where['ST'].copy()}
        other.next_choice = self.next_choice
        other.updated = self.updated
        other.newly_filled = self.newly_filled
        return other

    @classmethod
    def __get_orderlist(cls):
        def cal_order(ele):
            return len(cls.domains[ele])
        order = []
        for name, num in cls.shape_needed_num.items():
            order.extend([name] * num)
        order.sort(key = cal_order)
        return order

    @classmethod
    def setup(cls, nodes, edges1, edges2, pic):
        assert type(nodes) == set
        cls.shape_num = 7
        cls.g = Graph(nodes, edges1, edges2)
        cls.g.setup(pic)
        cls.domains = {'LT':cls.g.lts, 'MT':cls.g.mts, 'SQ':cls.g.sqs, 'PA':cls.g.pas, 'ST':cls.g.sts}
        cls.q = cls.__get_orderlist()
        cls.pic = pic

    def update(self):
        if self.updated:
            return
        del_sst = self.newly_filled.basics()
        self.sst_remain = [sst for sst in self.sst_remain if sst not in del_sst] 

        for shape_name, shape in self.shape_remain.items():
            self.shape_remain[shape_name][:] = [i for i in shape if not i.check_containing(del_sst)]
        self.updated = True
            

    def children(self):
        self.update()
        next_shape = self.q[self.next_choice]
        for block in self.shape_remain[next_shape]:
            next_child = self.fill_block(block)
            yield next_child
    
    def fill_block(self, block):
        child = copy.deepcopy(self)
        child.updated = False
        child.newly_filled = block
        child.next_choice = child.next_choice + 1
        child.where[block.name].append(block)
        #child.shape_remain[block.name].remove(block) 
        return child

        #don't update

    def is_finished(self):
        #don't have to update in advances
        if self.next_choice == self.shape_num:
            return True
        else:
            return False
        
    def visualise(self, pic, filename = None):
        if type(pic) == Picture:
            pic = pic.pic_ori
        color = pic.copy()
        if(len(color.shape) == 2):
            color = np.expand_dims(color, axis = 2)
            color = np.concatenate((color, color, color), axis = 2)     
        for item in self.where.items():
            for block in item[1]:
                points = np.array([n.coord for n in block.ordernodes()])
                cv2.fillPoly(color, [points], color = block.color)
        
        if filename is not None:
            cv2.imwrite(filename, color)
        return color

# a class to contain implementation of 'DFS' Search       
class Solve_Tangram:
    def __init__(self):
        self.solved = None
        self.open = [Status()]

    
    def search(self, view = False):
        results = []
        while self.open:
            searching = self.open.pop()
            if view:
                results.append(searching.visualise(searching.pic))
            if searching.is_finished():
                self.solved = searching
                break
            
            for child in searching.children():
                self.open.append(child)
        
        if self.solved is None:
            pass
        if view:
            return results           
        return self.solved




def process_generator(filename):
    results = []
    #Read File
    pic  = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    Pic = Picture(pic)
    nodes, edges1, edges2 = Pic.analysis_pic()
    pic  = Pic.visualise(pic, [])
    raw = Pic.visualise(pic, Pic.raw_keypoints)
    nodesp = Pic.visualise(pic, [n.coord for n in Pic.nodes])

    edge = Pic.visualise(nodesp, edges = edges1)
    edge = Pic.visualise(edge, edges = edges2)
    Status.setup(nodes, edges1, edges2, Pic)
    s = Solve_Tangram()
    results.append(pic)
    results.append(raw)
    results.append(nodesp)
    results.append(edge)
    results.extend(s.search(view = True))
    return results

#main method
def solve_result(filename):
    #Read File
    pic  = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    Pic = Picture(pic)
    nodes, edges1, edges2 = Pic.analysis_pic()
    Status.setup(nodes, edges1, edges2, Pic)
    s = Solve_Tangram()
    result = s.search()
    return result.visualise(result.pic)


if __name__ == '__main__':
    filename = 'tangram18.png'
    pic  = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    Pic = Picture(pic)
    nodes, edges1, edges2 = Pic.analysis_pic()
    raw = Pic.visualise(pic, Pic.raw_keypoints, filename = 'raw.png')
    nodesp = Pic.visualise(pic, [n.coord for n in Pic.nodes], filename = 'nodes.png')
    edge = Pic.visualise(nodesp, edges = edges1, filename = 'e1.png')
    edge = Pic.visualise(edge, edges = edges2, filename = 'all.png')

    Status.setup(nodes, edges1, edges2, Pic)

    s = Solve_Tangram()

    result = s.search()    
