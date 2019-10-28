import copy
import numpy as np
import cv2
from  math import sqrt
import math
# qiuqiu! pangqiuqiu :)

class Picture:
    def __init__(self, pic):
        self.pic_ori = pic
        self.edges1 = set([])
        self.nodes = set([])
        self.unit_lenth = self.cal_unit_l()
        self.raw_keypoints = []
        self.goodpoints = np.array([])
        self.p_candidates = None

    def analysis_pic(self):
        #assume the theta is default to 0 for now 
        #don't consider several graph in one pic for now
      
        #Find keypoints and exclude those fake ones and include unfound
        self.raw_keypoints = np.int0(cv2.goodFeaturesToTrack(self.pic_ori, 23, 0.17, self.unit_lenth // 4)) 
        self.raw_keypoints = [p.ravel() for p in self.raw_keypoints]
        self.valid_keypoints = Picture.find_valid(self.raw_keypoints, self.unit_lenth)
        
        for theta, points in self.valid_keypoints:
            self.find_nodes(points, theta)
        return self.nodes, self.edges1

    def cal_unit_l(self):
        #cal unit lenth
        h, w = self.pic_ori.shape
        black = 0
        for x in range(w):
            for y in range(h):
                if self.pic_ori[y][x] == 0:
                    black += 1
        return sqrt(black/45)     

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

        thetas = [0]
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
        self.ori_p = ori_p
        # if(ori_p[0] < 0):
        #     ori_p[0] = ori_p[0] + int(self.unit_lenth)
        # if(ori_p[1] < 0):
        #     ori_p[1] = ori_p[1] + int(self.unit_lenth) 

        #assert ori_p[0] >= 0
        #assert ori_p[1] >= 0  
        ori_p[0] = ori_p[0] - int(self.unit_lenth) * 4
        ori_p[1] = ori_p[1] - int(self.unit_lenth) * 4

        ori_p = R.dot(ori_p)[::-1]
        # don't wanna write compliated algorithm, thus simplify
        
        # h,w = self.pic_ori.shape
        # num_h = h // self.unit_lenth + 1 
        # num_w = w // self.unit_lenth + 1
        # num_h = int(num_h)
        # num_w = int(num_w)
        # if (num_h - 1) * int(self.unit_lenth) + ori_p[1] >= h:
        #     num_h = num_h - 1
        # if (num_w - 1) * int(self.unit_lenth) + ori_p[0] >= w:
        #     num_w = num_w - 1    
         
        h,w = self.pic_ori.shape
        num_h = h // self.unit_lenth + 8
        num_w = w // self.unit_lenth + 8
        num_h = int(num_h)
        num_w = int(num_w)            
        self.p_candidates = np.zeros((num_h ,num_w)).astype(np.bool)
        for i in range(num_h):
            for j in range(num_w):
                if self.point_valid(self.get_coord(i, j, ori_p, theta) + np.array([int(self.unit_lenth/2), int(self.unit_lenth/2)])):
                    self.p_candidates[i][j] = True
        #self.pack_up_nodes_and_edges(self.p_candidates, ori_p, theta) #pack up nodes and edges base on candidate and ori_p
        #self.p_candidates = self.p_candidates.tolist()
        return self.p_candidates

    def get_coord(self, i, j, ori_p, theta):
        ct = math.cos(theta / 180.*3.1415927)
        st = math.sin(theta / 180.*3.1415927)
        R = np.array([[ct, -st], [st, ct]])
        return (ori_p + R.dot(np.array([i, j]) * np.array([int(self.unit_lenth), int(self.unit_lenth)]))[::-1]).astype(np.int)

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
        # recommended output:  self.raw_keypoints = [] self.goodpoints = np.array([])
        # maybe web nodes?
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


class LL(BaseShape):
    name = 'LL'
    color = (100,100,100)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 

class L2(BaseShape):
    name = 'L2'
    color = (100,100,0)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 

class CO(BaseShape):
    name = 'CO'
    color = (0,100,100)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 

class L3(BaseShape):
    name = 'L3'
    color = (100,0,100)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 


class ST(BaseShape):
    name = 'ST'
    color = (150,50,100)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 

class TA(BaseShape):
    name = 'TA'
    color = (100,150,50)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 

class TT(BaseShape):
    name = 'TT'
    color = (100,150,200)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 

class ZZ(BaseShape):
    name = 'ZZ'
    color = (130,20,180)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 

class UU(BaseShape):
    name = 'UU'
    color = (200,180,20)
    def __init__(self, sst1, sst2, sst3, sst4, sst5):
        self.children = {'SST':[sst1, sst2, sst3, sst4, sst5]} 
    def ordernodes(self):
        return self.children['SST']
    def __eq__(self, other):
        if(type(other) != type(self)):
            return False
        for sst in other.children['SST']:
            if sst not in self.children['SST']:
                return False
        return True 


class Graph:
    def __init__(self, Pic):
        self.ssts = []#directed edge
        self.l2s = []#
        self.l3s = []
        self.cos = []
        self.lls = []
        self.sts = []
        self.tts = []
        self.zzs = []
        self.uus = []
        self.tas = []

    def setup(self, pic):
        self.p_candidates =  pic.p_candidates
        p_candidates = pic.p_candidates
        hn = p_candidates.shape[0]
        wn = p_candidates.shape[1]
        #ll l2 l3
        for h in range(1,hn-1):
            for w in range(1,wn-1):
                if p_candidates[h][w]:
                    if p_candidates[h][w-1] and p_candidates[h][w+1] and p_candidates[h-1][w+1]:
                        if p_candidates[h-2][w+1]:
                            self.lls.append(LL((h,w), (h,w-1), (h,w+1), (h-1 ,w+1), (h-2,w+1)))
                        if p_candidates[h+1][w-1]:
                            self.l3s.append(L3((h,w), (h,w-1), (h,w+1), (h-1 ,w+1), (h+1,w-1)))
                        if p_candidates[h][w-2]:
                            self.l2s.append(L2((h,w), (h,w-1), (h,w+1), (h-1 ,w+1), (h,w-2)))
                    if p_candidates[h-1][w] and p_candidates[h+1][w] and p_candidates[h+1][w+1]:
                        if p_candidates[h+1][w+2]:
                            self.lls.append(LL((h,w), (h-1,w), (h+1,w), (h+1 ,w+1), (h+1,w+2)))
                        if p_candidates[h-1][w-1]:
                            self.l3s.append(L3((h,w), (h-1,w), (h+1,w), (h+1 ,w+1), (h-1,w-1)))
                        if p_candidates[h-2][w]:
                            self.l2s.append(L2((h,w), (h-1,w), (h+1,w), (h+1 ,w+1), (h-2,w)))
                    if p_candidates[h][w+1] and p_candidates[h][w-1] and p_candidates[h+1][w-1]:
                        if p_candidates[h+2][w-1]:
                            self.lls.append(LL((h,w), (h,w+1), (h,w-1), (h+1 ,w-1), (h+2,w-1)))
                        if p_candidates[h-1][w+1]:
                            self.l3s.append(L3((h,w), (h,w+1), (h,w-1), (h+1 ,w-1), (h-1,w+1)))
                        if p_candidates[h][w+2]:
                            self.l2s.append(L2((h,w), (h,w+1), (h,w-1), (h+1 ,w-1), (h,w+2)))
                    if p_candidates[h+1][w] and p_candidates[h-1][w] and p_candidates[h-1][w-1]:
                        if p_candidates[h-1][w-2]:
                            self.lls.append(LL((h+1,w), (h-1,w), (h-1,w-1), (h-1 ,w-2), (h,w)))
                        if p_candidates[h+1][w+1]:
                            self.l3s.append(L3((h+1,w), (h-1,w), (h-1,w-1), (h+1 ,w+1), (h,w)))
                        if p_candidates[h+2][w]:
                            self.l2s.append(L2((h+1,w), (h-1,w), (h-1,w-1), (h+2 ,w), (h,w)))

        llt =[]
        for i in self.lls:
            if i not in llt:
                llt.append(i)
        self.lls = llt
        l3t =[]
        for i in self.l3s:
            if i not in l3t:
                l3t.append(i)
        self.l3s = l3t        
        #unchong 
        for h in range(1,hn-1):
            for w in range(1,wn-1):
                if p_candidates[h][w]:
                    if p_candidates[h][w+1] and p_candidates[h-1][w+1] and p_candidates[h+1][w-1] and p_candidates[h+1][w]:
                        self.cos.append(CO((h,w+1), (h-1,w+1), (h+1,w-1), (h+1,w), (h,w)))
                    if p_candidates[h][w-1] and p_candidates[h-1][w-1] and p_candidates[h+1][w+1] and p_candidates[h+1][w]:
                        self.cos.append(CO((h,w-1), (h-1,w-1), (h+1,w), (h+1,w+1), (h,w)))
                    if p_candidates[h-1][w] and p_candidates[h-1][w+1] and p_candidates[h+1][w-1] and p_candidates[h][w-1]:
                        self.cos.append(CO((h-1,w), (h-1,w+1), (h+1,w-1), (h,w-1), (h,w)))               
                    if p_candidates[h-1][w-1] and p_candidates[h-1][w] and p_candidates[h+1][w+1] and p_candidates[h][w+1]:
                        self.cos.append(CO((h-1,w-1), (h-1,w), (h+1,w+1), (h,w+1), (h,w)))      


        # #l2
        # for h in range(hn - 1):
        #     for w in range(wn - 1):
        #         if p_candidates[h][w]:
        #             if p_candidates[h][w+1] and p_candidates[h+1][w] and p_candidates[h+1][w+1]:
        #                 self.l2s.append(L2((h,w), (h+1,w), (h,w + 1), (h+1,w+1)))
        # #print('l2')
        # #l3
        # for h in range(hn - 2):
        #     for w in range(wn - 2):
        #         if p_candidates[h][w]:
        #             if p_candidates[h + 1][w] and p_candidates[h + 2][w]:
        #                 self.l3s.append(L3((h,w), (h + 1,w), (h+2,w)))
        #             if p_candidates[h][w + 1] and p_candidates[h][w + 2]:
        #                 self.l3s.append(L3((h,w), (h,w + 1), (h,w + 2)))
        # #print('l3')
        # #co
        # for h in range(hn):
        #     for w in range(wn):
        #         if p_candidates[h][w]:
        #             if h+1 < hn and p_candidates[h + 1][w]:
        #                 if w+1 < wn and p_candidates[h][w + 1]:
        #                     self.cos.append(CO((h,w), (h,w + 1), (h + 1,w)))
        #                 if w-1>=0 and p_candidates[h][w - 1]:
        #                     self.cos.append(CO((h,w), (h,w - 1), (h + 1,w)))
        #             if h-1 >=0 and p_candidates[h - 1][w]:
        #                 if w+1 < wn and p_candidates[h][w + 1]:
        #                     self.cos.append(CO((h,w), (h,w + 1), (h - 1,w)))
        #                 if w-1>=0 and p_candidates[h][w - 1]:
        #                     self.cos.append(CO((h,w), (h,w - 1), (h - 1,w)))
        # #print('co')
        #sts
        for h in range(1,hn-1):
            for w in range(1,wn-1):
                if p_candidates[h][w]:
                    if p_candidates[h + 1][w] and p_candidates[h - 1][w] and p_candidates[h][w - 1] and p_candidates[h][w + 1]:
                        self.sts.append(ST((h,w), (h,w - 1), (h,w + 1), (h + 1,w), (h - 1,w)))
        #print('st')
        #tas
        for h in range(1,hn-1):
            for w in range(1,wn-1):
                if p_candidates[h][w]:
                    if p_candidates[h - 1][w] and p_candidates[h -2][w] and p_candidates[h - 1][w + 1] and p_candidates[h - 2][w + 1]:
                        self.tas.append(TA((h,w), (h - 1,w), (h - 2,w), (h - 1,w + 1), (h - 2,w + 1)))
                    if p_candidates[h][w -1] and p_candidates[h][w-2] and p_candidates[h-1][w-1] and p_candidates[h-1][w-2]:
                        self.tas.append(TA((h,w), (h,w -1), (h,w-2), (h -1 ,w-1), (h-1,w-2)))
                    if p_candidates[h][w + 1] and p_candidates[h][w +2] and p_candidates[h + 1][w + 1] and p_candidates[h + 1][w +2]:
                        self.tas.append(TA((h,w), (h,w + 1), (h,w + 2), (h + 1,w + 1), (h + 1,w + 2)))
                    if p_candidates[h + 1][w] and p_candidates[h +2][w] and p_candidates[h + 1][w - 1] and p_candidates[h + 2][w - 1]:
                        self.tas.append(TA((h,w), (h + 1,w), (h + 2,w), (h + 1,w - 1), (h + 2,w - 1)))
        #print('ta')
        #tts
        for h in range(1,hn-1):
            for w in range(1,wn-1):
                if p_candidates[h][w]:
                    if p_candidates[h + 1][w] and p_candidates[h][w - 2] and p_candidates[h][w - 1] and p_candidates[h][w + 1]:
                        self.tts.append(TT((h + 1,w), (h,w - 2), (h,w- 1), (h,w), (h,w + 1)))
                    if p_candidates[h + 1][w] and p_candidates[h][w - 2] and p_candidates[h][w - 1] and p_candidates[h][w + 1]:
                        self.tts.append(TT((h,w), (h+1,w), (h,w-2), (h,w-1), (h,w+1)))
                    if p_candidates[h][w+1] and p_candidates[h][w +2] and p_candidates[h-1][w] and p_candidates[h][w - 1]:
                        self.tts.append(TT((h,w), (h,w +1), (h,w+2), (h-1,w), (h,w - 1)))
                    if p_candidates[h][w +1] and p_candidates[h+2][w] and p_candidates[h-1][w] and p_candidates[h+1][w]:
                        self.tts.append(TT((h,w), (h,w +1), (h+2,w), (h-1,w), (h+1,w)))
        #print('tt')
        #uus
        for h in range(1,hn-1):
            for w in range(1,wn-1):
                if p_candidates[h][w]:
                    if p_candidates[h-1][w+1] and p_candidates[h][w+1] and p_candidates[h-1][w-1] and p_candidates[h][w-1]:
                        self.uus.append(UU((h,w), (h-1,w+1), (h,w+1), (h-1,w-1), (h,w-1)))
                    if p_candidates[h+1][w+1] and p_candidates[h][w+1] and p_candidates[h+1][w-1] and p_candidates[h][w-1]:
                        self.uus.append(UU((h,w), (h+1,w+1), (h,w+1), (h+1,w-1), (h,w-1)))
                    if p_candidates[h+1][w] and p_candidates[h+1][w-1] and p_candidates[h-1][w] and p_candidates[h-1][w-1]:
                        self.uus.append(UU((h,w), (h+1,w), (h+1,w-1), (h-1,w), (h-1,w-1)))
                    if p_candidates[h+1][w] and p_candidates[h+1][w+1] and p_candidates[h-1][w] and p_candidates[h-1][w+1]:
                        self.uus.append(UU((h,w), (h+1,w), (h+1,w+1), (h-1,w), (h-1,w+1)))
        #print('uu')
        #zzs
        for h in range(1,hn-1):
            for w in range(1,wn-1):
                if p_candidates[h][w]:       
                    if p_candidates[h-1][w] and p_candidates[h-1][w+1] and p_candidates[h][w-1] and p_candidates[h][w-2]:
                        self.zzs.append(ZZ((h,w), (h-1,w), (h-1,w+1), (h,w-1), (h,w-2)))
                    if p_candidates[h][w+1] and p_candidates[h+1][w+1] and p_candidates[h-1][w] and p_candidates[h-2][w]:
                        self.zzs.append(ZZ((h,w), (h,w+1), (h+1,w+1), (h-1,w), (h-2,w)))
                    if p_candidates[h][w+1] and p_candidates[h][w+2] and p_candidates[h+1][w] and p_candidates[h+1][w-1]:
                        self.zzs.append(ZZ((h,w), (h,w+1), (h,w+2), (h+1,w), (h+1,w-1)))                    
                    if p_candidates[h+1][w] and p_candidates[h+2][w] and p_candidates[h][w-1] and p_candidates[h-1][w-1]:
                        self.zzs.append(ZZ((h,w), (h+1,w), (h+2,w), (h,w-1), (h-1,w-1)))  
        #print('zz')      


class Status:
    g = None
    q = None
    shape_num = None
    pic = None
    shape_needed_num = {'L2':1, 'L3':1, 'CO':1, 'ST':1, 'TA':1, 'TT':1,'ZZ':1,'UU':1, 'LL' : 1}
    
    def __init__(self,flag = True):
        assert type(self.g) == Graph
        self.l2_remain = self.g.l2s  
        self.ll_remain = self.g.lls#
        self.l3_remain = self.g.l3s
        self.co_remain = self.g.cos
        self.st_remain = self.g.sts
        self.ta_remain = self.g.tas
        self.tt_remain = self.g.tts
        self.zz_remain = self.g.zzs
        self.uu_remain = self.g.uus
        self.shape_remain = {'L2':self.l2_remain, 'L3':self.l3_remain, 'CO':self.co_remain,
                             'ST':self.st_remain, 'TA':self.ta_remain, 'TT':self.tt_remain,
                             'ZZ':self.zz_remain, 'UU':self.uu_remain, 'LL':self.ll_remain}
        self.where = {'L2':[], 'L3':[], 'CO':[], 'ST':[], 'TA':[],'TT':[],'ZZ':[],'UU':[],'LL':[]}
        self.next_choice = 0
        self.updated = True
        self.newly_filled = None
        self.sst_remain = set([])
        if flag:
            p_candidates = self.g.p_candidates
            hn = p_candidates.shape[0]
            wn = p_candidates.shape[1]
            for h in range(hn - 1):
                for w in range(wn - 1):
                    if p_candidates[h][w]:
                        self.sst_remain.add((h,w))
        


    def __deepcopy__(self, memo):
        other = Status(False)
        other.l2_remain = self.l2_remain.copy()
        other.l3_remain = self.l3_remain.copy()
        other.co_remain = self.co_remain.copy()
        other.st_remain = self.st_remain.copy()
        other.ta_remain = self.ta_remain.copy()
        other.uu_remain = self.uu_remain.copy()
        other.zz_remain = self.zz_remain.copy()
        other.tt_remain = self.tt_remain.copy()
        other.ll_remain = self.ll_remain.copy()
        other.sst_remain = self.sst_remain.copy()

        other.shape_remain = {'L2':other.l2_remain, 'L3':other.l3_remain, 'CO':other.co_remain,
                        'ST':other.st_remain, 'TA':other.ta_remain, 'TT':other.tt_remain,
                        'ZZ':other.zz_remain, 'UU':other.uu_remain, 'LL':other.ll_remain}

        other.where = {'L2':self.where['L2'].copy(), 'L3':self.where['L3'].copy(),
                     'CO':self.where['CO'].copy(), 'ST':self.where['ST'].copy(),
                     'TA':self.where['TA'].copy(), 'TT':self.where['TT'].copy(), 
                     'ZZ':self.where['ZZ'].copy(), 'UU':self.where['UU'].copy(),
                     'LL':self.where['LL'].copy()}
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
    def setup(cls, pic):
        cls.shape_num = 9
        cls.g = Graph(pic)
        cls.g.setup(pic)
        cls.domains = {'L2':cls.g.l2s, 'L3':cls.g.l3s, 'CO':cls.g.cos, 'ST':cls.g.sts, 'TA':cls.g.tas,
                        'TT':cls.g.tts,'ZZ':cls.g.zzs,'UU':cls.g.uus, 'LL':cls.g.lls}
        cls.q = cls.__get_orderlist()
        cls.pic = pic

    def dead(self):
        #return False
        self.update()
        a = self.sst_remain
        for key,it in self.shape_remain.items():
            for i in it:
                a = a - set(i.children['SST'])
            if len(a) == 0:
                return False
        
        return True

    def update(self):
        if self.updated:
            return
        del_sst = self.newly_filled.basics()
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
        child.sst_remain = child.sst_remain - set(block.children['SST'])
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
                for sq in block.children['SST']:
                    h,w = sq
                    points = np.array([self.pic.get_coord(h, w, self.pic.ori_p, 0), 
                                    self.pic.get_coord(h+1, w, self.pic.ori_p, 0),
                                    self.pic.get_coord(h+1, w+1, self.pic.ori_p, 0),
                                    self.pic.get_coord(h, w+1, self.pic.ori_p, 0)])
                    cv2.fillPoly(color, [points], color = block.color)
        
        if filename is not None:
            cv2.imwrite(filename, color)
        return color
        
class Solve_Tangram:
    def __init__(self):
        self.solved = None
        self.open = [Status()]

    
    def search(self, view = False):
        import time
        flag = 0
        results = []
        uu = 0
        while self.open:
            # uu = uu + 1
            searching = self.open.pop()
            # print(uu)
            # # print(time.time() - flag)
            # # while time.time() - flag<0.5:
            # #     pass
            # # flag = time.time()
            #cv2.imwrite('searching.png',searching.visualise(searching.pic)  )
            if view:
                results.append(searching.visualise(searching.pic))
            if searching.is_finished():
                self.solved = searching
                break
            if searching.dead():
                continue
            for child in searching.children():
                self.open.append(child)
        
        if self.solved is None:
            return None
        if view:
            return results           
        return self.solved




def process_generator(filename):
    results = []
    #Read File
    pic  = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    Pic = Picture(pic)
    Pic.analysis_pic()
    pic  = Pic.visualise(pic, [])
    raw = Pic.visualise(pic, Pic.raw_keypoints)
    Status.setup(Pic)
    s = Solve_Tangram()
    results.append(pic)
    results.append(raw)
    results.extend(s.search(view = True))
    return results

def solve_result(filename):
    #Read File
    pic  = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    Pic = Picture(pic)
    Pic.analysis_pic()
    Status.setup(Pic)
    s = Solve_Tangram()
    result = s.search()
    return result.visualise(result.pic)


if __name__ == '__main__':
    filename = '4.png'
    pic  = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    Pic = Picture(pic)
    Pic.analysis_pic()
    #print('pic')
    # raw = Pic.visualise(pic, Pic.raw_keypoints, filename = 'raw.png')
    # nodesp = Pic.visualise(pic, [n.coord for n in Pic.nodes], filename = 'nodes.png')
    # edge = Pic.visualise(nodesp, edges = edges1, filename = 'e1.png')
    # edge = Pic.visualise(edge, edges = #edges2, filename = 'all.png')
    Status.setup(Pic)

    s = Solve_Tangram()

    result = s.search() 
    cv2.imwrite('13.png',result.visualise(result.pic)  )