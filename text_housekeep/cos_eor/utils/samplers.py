import magnum as mn
from matplotlib.path import Path
import numpy as np

MAX_HEIGHT_DIFF = 0.15

def def_is_legal(start_pos, pos):
    diff_y = abs(pos[1] - start_pos[1])
    if diff_y > MAX_HEIGHT_DIFF:
        return False
    return True

class ObjSampler(object):
    def sample(self, old_pos, obj_idx):
        raise ValueError()

    def should_stabilize(self):
        return False

    def should_add_offset(self):
        return True

    def reset_balance(self):
        pass

    def is_legal(self, start_pos, pos):
        return def_is_legal(start_pos, pos)

    def set_sim(self, sim):
        self.sim = sim

class PolySurface(ObjSampler):
    def __init__(self, height, poly, height_noise=0.0, trans=None):
        self.height = height
        if trans is not None:
            self.poly = [trans.transform_point(mn.Vector3(p[0], height, p[1])) for p in poly]
            self.poly = [[x[0], x[2]] for x in self.poly]
        else:
            self.poly = poly
        self.height_noise = height_noise

    def should_stabilize(self):
        return True

    def sample(self, old_pos, obj_idx):
        size = 1000
        extent = 3

        # Draw a square around the average of the points
        avg = np.mean(np.array(self.poly), axis=0)
        x, y = np.meshgrid(
                np.linspace(avg[0]-extent,avg[0]+extent,size),
                np.linspace(avg[1]-extent,avg[1]+extent,size))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T

        p = Path(self.poly) # make a polygon
        grid = p.contains_points(points)
        points = points.reshape(size**2,2)
        valid_points = points[grid == True]
        self.p = p

        use_x, use_y = valid_points[np.random.randint(len(valid_points))]
        return [use_x,self.height + np.random.uniform(0, self.height_noise),use_y]

    def is_legal(self, start_pos, pos):
        pos_2d = [pos[0], pos[2]]
        #if not self.p.contains_point(pos_2d):
        #    return False
        diff_y = abs(pos[1] - start_pos[1])
        if diff_y > (MAX_HEIGHT_DIFF + self.height_noise):
            #print('Y diff', diff_y)
            return False
        return True
