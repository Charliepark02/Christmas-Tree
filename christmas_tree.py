"""Run in the visual conda environment. Q/Esc quits, R starts over."""
import subprocess
import tempfile
import wave
from pathlib import Path

import math
import time
from collections import deque

import cv2
import numpy as np

W, H, GROUND = 960, 640, 592
TAU = 2 * math.pi


class HandRotation:
    """Integrate rotation of a palm-fixed frame, independent of fingertip travel."""
    def __init__(self):
        self.anchor = None
        self.filtered = None

    def advance(self, points):
        across = points[5] - points[17]
        forward = points[9] - points[0]
        normal = np.cross(across, forward)
        if min(np.linalg.norm(across), np.linalg.norm(normal)) < 1e-7:
            self.__init__()
            return 0.
        across /= np.linalg.norm(across)
        normal /= np.linalg.norm(normal)
        basis = np.column_stack((across, np.cross(normal, across), normal))
        if self.anchor is None:
            self.anchor = self.filtered = basis
            return 0.
        u, _, v = np.linalg.svd(.55*self.filtered + .45*basis)
        self.filtered = u @ v
        cosine = (np.trace(self.anchor.T @ self.filtered) - 1) / 2
        angle = math.acos(np.clip(cosine, -1, 1))
        if angle < math.radians(10):
            return 0.
        self.anchor = self.filtered
        return angle if angle < math.radians(80) else 0.


class Gesture:
    def __init__(self):
        self.rotation = HandRotation()
        self.pinch = deque()
        self.fist_since = None
        self.release_until = -100.
        self.open_since = None
        self.thumb_since = None
        self.last = -100.

    def read(self, p, side, now):
        if now - self.last > .35:
            self.__init__()
        self.last = now
        scale = max(1e-6, np.linalg.norm(p[5] - p[17]))
        distance = lambda a, b: np.linalg.norm(p[a] - p[b]) / scale
        extended = [distance(i, 0) > distance(i - 2, 0) + .20 for i in (8, 12, 16, 20)]
        gap = distance(4, 8)
        self.spread = all(extended) and gap > .8
        if sum(extended) <= 2:
            self.gathered = now
        palm = (p[0] + p[9]) / 2
        thumb_up = (not any(extended) and distance(4, 5) > .95
                    and p[2,1]-p[4,1] > .65*scale
                    and p[3,1]-p[4,1] > .16*scale
                    and abs(p[2,0]-p[4,0]) < p[2,1]-p[4,1])
        if thumb_up:
            if self.thumb_since is None:
                self.thumb_since = now
            self.release_until = -100.
            self.fist_since = None
            self.open_since = None
            self.rotation = HandRotation()
            self.pinch.clear()
            if now-self.thumb_since >= .3:
                return 'greeting'
            return None
        self.thumb_since = None
        tight_fist = (not any(extended) and distance(4, 5) < .85 and
                      all(np.linalg.norm(p[i]-palm) < .92*scale and
                          distance(i, 0) < distance(i-2, 0)
                          for i in (8, 12, 16, 20)))
        if tight_fist:
            if self.fist_since is None:
                self.fist_since = now
            if now - self.fist_since >= .38:
                self.release_until = now + .85
        else:
            self.fist_since = None
        if self.spread and now <= self.release_until:
            if self.open_since is None:
                self.open_since = now
            if now - self.open_since >= .12:
                self.release_until = -100.
                self.pinch.clear()
                self.rotation = HandRotation()
                return 'star'
        else:
            self.open_since = None
        # Repeated thumb/index rubs with curled outer fingers mimic sprinkling.
        while self.pinch and now - self.pinch[0][0] > 1.6:
            self.pinch.popleft()
        if not extended[-1] and gap < .75:
            phase = True if gap < .29 else False if gap > .43 else None
            if phase is not None and (not self.pinch or self.pinch[-1][1] != phase):
                self.pinch.append((now, phase))
            if len(self.pinch) >= 4:
                self.pinch.clear()
                self.rotation = HandRotation()
                return 'snow'
        if extended[0] and sum(extended[1:]) <= 1 and gap > .65:
            return ('tree' if side == 'Right' else 'decor', self.rotation.advance(p))
        self.rotation = HandRotation()


def glint(img, x, y, radius, color):
    """Add a soft halo and a narrow four-point diffraction sparkle."""
    x,y,r = round(x),round(y),max(2,round(radius))
    h,w = img.shape[:2]
    x0,x1 = max(0,x-r),min(w,x+r+1)
    y0,y1 = max(0,y-r),min(h,y+r+1)
    if x0 >= x1 or y0 >= y1:
        return
    yy,xx = np.ogrid[y0-y:y1-y,x0-x:x1-x]
    halo = np.exp(-4*(xx*xx+yy*yy)/(r*r))*.32
    rays = np.maximum(np.maximum(0,1-abs(xx)/1.4)*(1-abs(yy)/r)**2,
                      np.maximum(0,1-abs(yy)/1.4)*(1-abs(xx)/r)**2)
    light = ((halo+rays)[:,:,None]*color).clip(0,255).astype(np.uint8)
    patch = img[y0:y1,x0:x1]
    cv2.add(patch,light,dst=patch)


class SleighRide:
    """One right-to-left flight with three parcels released along its path."""
    duration = 6.
    parcels = ((115, 22, (80,65,205)), (20, 34, (130,147,35)),
               (-103, 27, (177,90,143)))

    def __init__(self, start):
        self.start = start
        self.landed = set()
        self.impacts = {}

    def position(self, age):
        return (W+140 - (W+330)*age/self.duration, 137+13*math.sin(age*1.25))

    def draw(self, img, now, bank):
        age = now-self.start
        if age < self.duration:
            x,y = self.position(age)
            # A braided comet tail follows the runners along their recent flight path.
            for k in range(52):
                lag = k/51
                tx,ty = self.position(max(0,age-.75*lag))
                phase = 12*lag+age*5
                px = tx+55+8*math.sin(k*23)
                py = ty+16+(5+11*lag)*math.sin(phase+k%2*math.pi)
                tint = (118,210,255) if k%2 else (242,167,225)
                glint(img,px,py,2+4*(1-lag),tuple(c*(1-lag)**.8 for c in tint))
            self.draw_team(img, round(x), round(y), age)
        for offset,size,color in self.parcels:
            target = W//2 + offset
            drift = 45
            release = (W+140+30-target-drift)*self.duration/(W+330)
            elapsed = age-release
            if elapsed < 0:
                continue
            x,y = self.position(release)
            # Parcels keep a little sideways momentum as they leave the sleigh.
            cx = round(target+drift*(1-min(1.,elapsed/1.15))**2)
            floor = GROUND-bank[cx]-size/2
            cy = min(floor, y+8+32*elapsed+215*elapsed**2)
            if cy >= floor:
                self.landed.add(size)
                self.impacts.setdefault(size,now)
            else:
                for k in range(5):
                    glint(img,cx+5*math.sin(7*elapsed+k),cy-9*k,5-k*.6,
                          (110-k*16,190-k*27,245-k*35))
            angle = 12*math.sin(elapsed*5)*min(1,max(0,(floor-cy)/45))
            def panel(left,top,right,bottom,paint):
                corners = np.array([[left,top],[right,top],[right,bottom],[left,bottom]])*size
                transform = cv2.getRotationMatrix2D((0,0),angle,1)[:,:2]
                pts = np.rint(corners @ transform.T + [cx,cy]).astype('int32')
                cv2.fillConvexPoly(img,pts,paint,cv2.LINE_AA)
            panel(-.5,-.45,.5,.5,color)
            panel(-.54,-.51,.54,-.29,tuple(min(255,c+25) for c in color))
            panel(-.085,-.51,.085,.5,(110,211,252))
            for direction in (-1,1):
                cv2.ellipse(img,(cx+direction*size//8,round(cy-size*.56)),
                            (size//7,3),direction*25-angle,0,360,(121,217,253),1,cv2.LINE_AA)

            shimmer = (.5+.5*math.sin(3*elapsed+size))**4
            glint(img,cx+2,cy-size*.50,3+8*shimmer,(110,200,250))
            if size in self.impacts:
                u = (now-self.impacts[size])/.75
                if 0 <= u < 1:
                    for k in range(17):
                        a = math.pi*(k+.5)/17
                        reach = (32+11*math.sin(17*k))*u
                        glint(img,cx+reach*math.cos(a),floor+size/2-reach*math.sin(a)+12*u*u,
                              2+4*(1-u),tuple(v*(1-u) for v in (160,218,255)))

    @staticmethod
    def draw_team(img,x,y,age):
        aa = cv2.LINE_AA
        def oval(dx,dy,axes,color,tilt=0):
            cv2.ellipse(img,(x+dx,y+dy),axes,tilt,0,360,color,-1,aa)
        def stroke(points,color,width):
            cv2.polylines(img,[np.array(points,np.int32)+[x,y]],False,color,width,aa)
        def shape(points,color):
            cv2.fillPoly(img,[(np.array(points)+[x,y]).astype('int32')],color,aa)
        gold, red, fur = (79,187,239),(49,48,186),(66,111,162)
        # Rudolph leads. Staggered hoof beats give the airborne gallop a rhythm.
        for k,hip in enumerate((-79,-72,-49,-43)):
            kick = round(9*math.sin(6*age+k*1.7))
            stroke([(hip,0),(hip+kick,10),(hip+kick-6,16)],fur,3)
        oval(-61,-4,(24,11),fur)
        oval(-82,-14,(8,14),fur,-30)
        oval(-92,-23,(12,7),fur,-12)
        oval(-100,-21,(6,4),fur)
        oval(-104,-22,(4,4),(76,67,255))
        oval(-93,-25,(2,2),(25,26,34))
        oval(-83,-29,(6,3),fur,-30)
        for off in (0,6):
            stroke([(-91+off,-29),(-94+off,-36),(-91+off,-45)],gold,2)
            stroke([(-94+off,-36),(-100+off,-41)],gold,2)
        stroke([(-83,-13),(-61,-8),(-28,0),(0,-7)],gold,1)
        # Santa, his sack, and a curved red sleigh on gold runners.
        oval(44,-7,(14,15),(47,94,143),15)
        oval(16,-10,(12,16),red)
        oval(12,-28,(9,10),(153,194,233))
        oval(8,-23,(8,7),(224,235,243))
        shape([(2,-34),(13,-49),(25,-34)],red)
        stroke([(3,-34),(20,-34)],(228,239,249),4)
        oval(13,-48,(3,3),(230,240,250))
        stroke([(14,-12),(4,-10),(-2,-8)],red,5)
        shape([(-14,-8),(-6,-3),(29,-2),(57,-13),(53,7),(43,13),(3,13),(-8,6)],red)
        stroke([(-14,-8),(-6,-3),(29,-2),(57,-13)],gold,2)
        for leg in (5,42):
            stroke([(leg,12),(leg+3,19)],gold,2)
        stroke([(-18,12),(-13,18),(-5,20),(53,20),(61,17)],gold,3)


class Scene:
    def __init__(self):
        self.rng = np.random.default_rng(24)
        r = self.rng
        self.tree = self.decor = 0.
        self.lights_at = None
        self.openings = {}
        self.star = None
        self.greeting_at = None
        self.ride = None
        self.swipe_path = deque()
        self.spin = 0.
        self.frost = 0.
        self.snowing = False
        self.motion = {layer: deque() for layer in ('tree','decor')}
        self.started = {}
        self.flakes = []
        self.bank = np.zeros(W)
        self.credit = 0.
        position = np.arange(W)
        self.bank_limit = 30 + 5*np.sin(position/112) + 3*np.cos(position/47)
        self.deposit = .8*np.exp(-(np.arange(-8,9)/4.5)**2)
        # Overlapping tapered whorls, each with a scalloped lower edge.
        tier = r.integers(0, 9, 52000)
        u = r.random(len(tier)) ** .55
        a = r.uniform(0, TAU, len(tier))
        self.height = 1 - (.08*tier + .24*u + .023*u*np.sin(11*a+tier))
        self.angle = a
        self.radius = (24 + tier*21)*u*(.67 + .33*r.random(len(tier)))
        self.shade = r.uniform(.32, 1, len(tier))
        self.frost_sites = r.permutation(np.flatnonzero((u > .87) & (self.radius > (24+tier*21)*.78)))[:380]
        # Fill the exposed underside with staggered needles. Leave the trunk visible below.
        skirt = np.random.default_rng(825)
        count = 19500
        angle = skirt.uniform(0, TAU, count)
        reach = skirt.uniform(.56, 1., count)
        radius = 192*reach*skirt.uniform(.64, 1.02, count)
        height = (.36-.24*reach-.023*reach*np.sin(11*angle+8)
                  -skirt.uniform(.006,.049,count))
        self.angle = np.concatenate((self.angle, angle))
        self.radius = np.concatenate((self.radius, radius))
        self.height = np.concatenate((self.height, height))
        self.shade = np.concatenate((self.shade, skirt.uniform(.32,.90,count)))
        self.sky = np.zeros((H,W,3),np.uint8)
        self.sky[:] = np.linspace((19,14,10),(34,24,16),H)[:,None,:]
        for x,y in zip(r.integers(W,size=550),r.integers(GROUND,size=550)):
            self.sky[y,x] = r.integers(34,96)
        self.bulbs = r.uniform(0,1,160)
        self.palette = [(73,75,255),(40,190,255),(255,130,65),(200,64,255),(110,250,150)]

    def opening(self, side, hand, now):
        if self.lights_at is not None:
            return
        if hand.spread and now - getattr(hand, 'gathered', -100.) < .65:
            self.openings.setdefault(side, now)
        elif not hand.spread:
            self.openings.pop(side, None)
        if len(self.openings) == 2 and min(self.openings.values()) >= now - .45:
            self.lights_at = now

    def swipe(self, now, palm):
        path = self.swipe_path
        if palm is None or (path and now-path[-1][0] > .2):
            path.clear()
        if palm is None:
            return
        path.append((now,*palm))
        while path and now-path[0][0] > .8:
            path.popleft()
        for start,x,y in path:
            dx = x-palm[0]
            if now-start >= .1 and dx >= .23 and abs(y-palm[1]) < .55*dx:
                if self.ride is None or now-self.ride.start >= self.ride.duration:
                    self.ride = SleighRide(now)
                path.clear()
                return

    def draw_base(self, img, t):
        age = t - self.lights_at
        growth = min(1., max(0., age / .7))
        cx = W // 2
        # Each pool has a soft falloff and an independently pulsing core.
        band = img[GROUND-12:GROUND+34, cx-155:cx+156]
        yy, xx = np.ogrid[-12:34, -155:156]
        glow = np.zeros_like(band, dtype=np.float32)
        for i, color in enumerate(self.palette):
            center = (i-2)*45
            pulse = .78 + .22*math.sin(age*2+i)
            d = ((xx-center)/34)**2 + ((yy-10)/7)**2
            glow += (growth*pulse*(.55*np.exp(-d)+.48*np.exp(-6*d)))[:,:,None]*np.array(color)
        band[:] = np.minimum(255, band + glow)
        growth = self.tree
        if growth <= 0:
            return
        # Flattened, tapering buttress roots fan out at different depths.
        for reach, end_y, bend in [(-99,-4,.8),(86,-6,1.9),(-132,4,2.4),
                                   (121,6,.4),(-52,12,1.2),(46,15,2.7)]:
            u = np.linspace(0, growth, 42)
            x = cx + reach*u**1.35 + 2*u*np.sin(9*u+bend)
            y = GROUND-56 + (56+end_y)*(1-(1-u)**3) + 3*u*np.sin(8*u+bend)
            width = (1-u)**1.2*9.5 + .5
            edge = np.column_stack((x-width*.55, y+width))
            ridge = np.column_stack((x+width*.55, y-width))
            cv2.fillPoly(img,[np.vstack((edge,ridge[::-1])).astype('int32')],
                         (43,76,103),cv2.LINE_AA)
            cv2.polylines(img,[ridge.astype('int32')],False,(69,114,151),1,cv2.LINE_AA)
            for fork, drop in ((.5,-6),(.72,8)):
                if growth <= fork:
                    continue
                idx = round(fork/growth*(len(u)-1))
                v = np.linspace(0,min(1.,(growth-fork)/.25),16)
                twig = np.column_stack((x[idx]+reach*.27*v,
                                        y[idx]+drop*v+2*np.sin(v*4)))
                cv2.polylines(img,[twig.astype('int32')],False,(58,96,122),1,cv2.LINE_AA)

    def trigger(self, event, t):
        if event == 'greeting':
            if self.greeting_at is None:
                self.greeting_at = t
        elif event == 'snow':
            self.snowing = True
        elif event == 'star' and self.tree > 0 and self.star is None:
            self.star = t
        elif isinstance(event, tuple) and self.lights_at is not None:
            layer, radians = event
            if layer in self.started or (layer == 'decor' and self.tree == 0):
                return
            history = self.motion[layer]
            while history and t-history[0][0] > 2.8:
                history.popleft()
            if radians > 0:
                history.append((t,radians))
                if sum(angle for _,angle in history) >= math.radians(55):
                    self.started[layer] = t

    def draw_growth_ring(self, img, age):
        if not 0 <= age < 1.8:
            return
        scatter = max(0., (age-1.45)/.35)
        height = self.tree + .035*math.sin(math.pi*self.tree)
        radius = 4 + 194*(1-height)
        phase = np.arange(220)*TAU/220
        angle = phase + 3.6*age
        jitter = np.sin(phase*37+1)
        x = W/2+(radius+jitter*2)*np.cos(angle)+scatter*23*np.sin(phase*17)
        y = (GROUND-5-481*height + .14*radius*np.sin(angle)
             + 1.3*np.sin(phase*23) - scatter*17*np.cos(phase*13))
        fade = min(1., age/.09)*(1-scatter)
        strength = fade*(.17+.83*(.5+.5*np.cos(phase*3))**2)
        hues = np.array([(200,230,255),(214,143,245),(189,255,145),(255,218,127)])
        paint = (hues[(np.arange(220)//18)%4]*strength[:,None]).astype(int)
        for px,py,color in zip(x,y,paint):
            ix,iy = round(px),round(py)
            img[iy,ix] = np.minimum(255,img[iy,ix]+color)

    def draw(self,t,dt):
        img = self.sky.copy()
        for layer, start in self.started.items():
            duration = 1.45 if layer == 'tree' else 1.15
            fraction = min(1.,max(0.,(t-start)/duration))
            setattr(self,layer,fraction*fraction*(3-2*fraction))
        self.spin += .23*dt
        spin = self.spin + self.tree*TAU*.48
        if self.tree > 0:
            growth = self.tree
            a = self.angle + spin
            depth = np.sin(a)
            x = (W/2+self.radius*np.cos(a)).astype(int)
            y = (GROUND-16-470*self.height + .16*self.radius*depth).astype(int)
            visible = self.height <= growth
            light = self.shade*(.65+.35*depth)
            colors = np.column_stack((32+85*light,45+182*light,12+95*light))
            leaf_y, leaf_x = y[visible], x[visible]
            needles = colors[visible].astype(np.uint8)
            # A shaded pixel below each tip fills the canopy without flattening the tiers.
            img[leaf_y+1,leaf_x] = (needles * .62).astype(np.uint8)
            img[leaf_y,leaf_x] = needles
            # Fixed leaf sites rotate with the canopy; only their brightness oscillates.
            specks = np.arange(71,len(x),283)
            pulse = (.5+.5*np.sin(2.8*t+specks*1.618))**8
            for j in np.flatnonzero(visible[specks] & (depth[specks] > 0) & (pulse > .38)):
                n = specks[j]
                glint(img,x[n],y[n],3+5*pulse[j],(110*pulse[j],219*pulse[j],158*pulse[j]))
            if growth < 1:
                rim = specks[(self.height[specks] <= growth) &
                             (self.height[specks] > growth-.09)]
                for n in rim:
                    glint(img,x[n],y[n],8,(154,231,209))
            if growth >= 1 and self.decor >= 1 and self.star is not None and self.flakes:
                near_tree = sum(100 < fy < GROUND-25 and abs(fx-W/2) < (fy-96)*.38
                                for fx,fy,*_ in self.flakes)
                self.frost = min(len(self.frost_sites), self.frost + dt*min(18,near_tree*.23))
            sites = self.frost_sites[:int(self.frost)]
            front = sites[depth[sites] > 0]
            img[y[front],x[front]] = (205,210,202)
            img[y[front],x[front]+1] = (155,170,158)
            sweep = None
            if self.greeting_at is not None:
                elapsed = t-self.greeting_at
                if 0 <= elapsed < 1.55:
                    sweep = -.15 + 1.3*elapsed/1.55
            if self.decor > 0:
                for i,h in enumerate(self.bulbs):
                    if h > min(growth,self.decor):
                        continue
                    angle = TAU*5*h + spin
                    rad = 185*(1-h)
                    if math.sin(angle) < -.18:
                        continue
                    pos = (int(W/2+rad*math.cos(angle)),int(GROUND-16-470*h+.16*rad*math.sin(angle)))
                    color = self.palette[i%5]
                    flash = max(0.,1-abs(h-sweep)/.15) if sweep is not None else 0.
                    twinkle = (.5+.5*math.sin(t*3.1+i*2.4))**5
                    ignition = max(0.,1-(self.decor-h)/.11) if self.decor < 1 else 0.
                    energy = max(twinkle,ignition,flash)
                    if energy > .32:
                        glint(img,*pos,5+11*energy,tuple(c*energy for c in color))
                    if flash:
                        color = tuple(round(c+(255-c)*flash) for c in color)
                    cv2.circle(img,pos,3+round(3*flash),color,-1,cv2.LINE_AA)
                    cv2.circle(img,pos,1,(225,245,255),-1)
            if self.star is not None and growth >= 1:
                cx,cy = W//2,GROUND-495
                pulse = 1+.13*math.sin(4*(t-self.star))
                angles = np.arange(10)*math.pi/5-math.pi/2
                radius = np.tile([19,8],5)*pulse
                pts = np.column_stack((cx+radius*np.cos(angles),cy+radius*np.sin(angles))).astype('int32')
                for j in range(8):
                    a = j*TAU/8 + .13*t
                    cv2.line(img,(cx,cy),(int(cx+35*pulse*math.cos(a)),int(cy+35*pulse*math.sin(a))),(65,150,210),1,cv2.LINE_AA)
                cv2.fillPoly(img,[pts],(115,225,255),cv2.LINE_AA)
        if self.snowing:
            self.credit += dt*62
            for _ in range(int(self.credit)):
                depth = self.rng.random()
                self.flakes.append((self.rng.uniform(0,W), -3., depth,
                                    self.rng.uniform(0,TAU), self.rng.uniform(.8,1.2)))
            self.credit %= 1
        falling = []
        wind = 12*math.sin(.43*t) + 7*math.sin(.17*t)
        for x,y,depth,phase,variation in self.flakes:
            sway = math.sin(t*variation + phase)
            x = (x + dt*(wind + 22*sway)*(.4+depth)) % W
            y += dt*(39 + 103*depth)*variation*(1+.09*math.cos(phase+t))
            i = int(x)
            if y >= GROUND-self.bank[i]:
                lo,hi = max(0,i-8),min(W,i+9)
                self.bank[lo:hi] = np.minimum(self.bank_limit[lo:hi],
                    self.bank[lo:hi]+self.deposit[lo-i+8:hi-i+8])
            else:
                falling.append((x,y,depth,phase,variation))
                # Subpixel ellipses with dim edges and varied focus/depth.
                center = (round(x*16),round(y*16))
                radius = .48+2.15*depth**2
                value = int(64+142*depth+9*sway)
                for size,brightness in ((1.5,.33),(.83,.78),(.45,1.)):
                    axes = (max(3,round(16*radius*size)),max(3,round(20*radius*size)))
                    tone = int(value*brightness)
                    cv2.ellipse(img,center,axes,22*sway,0,360,(tone+8,tone+4,tone),
                                -1,cv2.LINE_AA,4)
        self.flakes = falling
        ridge = np.column_stack((np.arange(W),GROUND-self.bank)).astype('int32')
        cv2.fillPoly(img,[np.vstack((ridge,[[W-1,H],[0,H]])).astype('int32')],(74,65,55))
        floor = np.column_stack((np.arange(W-1,-1,-1), np.full(W,GROUND)))
        cv2.fillPoly(img,[np.vstack((ridge,floor)).astype('int32')],(183,177,164),cv2.LINE_AA)
        cv2.polylines(img,[ridge],False,(213,209,198),1,cv2.LINE_AA)
        if self.lights_at is not None:
            self.draw_base(img,t)
        if 'tree' in self.started:
            self.draw_growth_ring(img,t-self.started['tree'])
        if self.ride is not None:
            self.ride.draw(img,t,self.bank)
        if self.greeting_at is not None:
            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
            message = 'Merry Christmas'
            (width,height),_ = cv2.getTextSize(message,font,1.65,2)
            origin = ((W-width)//2, 22+height)
            header = img[:origin[1]+18]
            lettering = header.copy()
            cv2.putText(lettering,message,origin,font,1.65,(27,34,48),5,cv2.LINE_AA)
            cv2.putText(lettering,message,origin,font,1.65,(144,217,250),2,cv2.LINE_AA)
            alpha = min(1., max(0., (t-self.greeting_at)/.95))
            alpha = alpha*alpha*(3-2*alpha)
            cv2.addWeighted(lettering,alpha,header,1-alpha,0,dst=header)
        return cv2.addWeighted(img,1,cv2.GaussianBlur(img,(0,0),3),.45,0)


class WinterAudio:
    """Synthesize local WAVs once and play them off the camera thread."""
    rate = 24000

    def __init__(self):
        self.folder = tempfile.TemporaryDirectory(prefix='christmas-audio-')
        self.players = {}
        self.played_layers = set()
        self.snow_started = False
        self.star_played = False
        self.ride = None
        self.landed = set()
        self.build_clips()

    def save(self, name, samples):
        pcm = (np.clip(samples,-1,1)*32767).astype('<i2')
        with wave.open(str(Path(self.folder.name)/(name+'.wav')), 'wb') as out:
            out.setparams((1 if pcm.ndim == 1 else 2,2,self.rate,0,'NONE',''))
            out.writeframes(pcm.tobytes())

    def build_clips(self):
        rng = np.random.default_rng(1225)
        def clock(seconds):
            return np.arange(round(self.rate*seconds))/self.rate
        snow_rng = np.random.default_rng(101)
        def powder(seconds, clusters):
            samples = np.zeros(round(self.rate*seconds))
            for onset, span, grains, level in clusters:
                for _ in range(grains):
                    offset = snow_rng.uniform(0, span)
                    start = round((onset+offset)*self.rate)
                    u = clock(snow_rng.uniform(.014,.045))
                    noise = snow_rng.normal(size=len(u))
                    soft = np.convolve(noise,np.hanning(13)/6,mode='same')
                    soft -= np.convolve(soft,np.ones(49)/49,mode='same')
                    envelope = np.sin(np.linspace(0,np.pi,len(u)))**2
                    samples[start:start+len(u)] += level*(1-.65*offset/span)*soft*envelope
            return samples
        self.save('snow_start',powder(.85,[(.02,.62,105,.14)]))
        clusters = [(t,snow_rng.uniform(.12,.32),18,.055)
                    for t in (.8,3.7,5.1,8.5,12.7,14.4,18.1)]
        self.save('snow_settle',powder(20,clusters))
        t = clock(1.9)
        chime = np.zeros_like(t)
        for onset,freq in [(0,784),(.13,988),(.27,1319)]:
            u = np.maximum(0,t-onset)
            chime += .16*(1-np.exp(-95*u))*np.exp(-3.6*u)*(
                np.sin(TAU*freq*u)+.28*np.sin(TAU*freq*2.01*u))
        self.save('star',chime)
        t = clock(SleighRide.duration)
        beat = t % .31
        envelope = (1-np.exp(-180*beat))*np.exp(-17*beat)
        bells = .06*envelope*np.sin(np.pi*t/SleighRide.duration)**1.2*sum(
            np.sin(TAU*f*t) for f in (1270,1741,2257,3109))
        pan = t/SleighRide.duration*np.pi/2
        self.save('sleigh',np.column_stack((bells*np.sin(pan),bells*np.cos(pan))))
        t = clock(.28)
        for _,size,_ in SleighRide.parcels:
            freq = 4900/size
            knock = np.exp(-19*t)*np.minimum(1,220*t)*(
                .36*np.sin(TAU*(freq*t+1.2*(1-np.exp(-35*t)))) +
                .08*rng.normal(size=len(t))*np.exp(-45*t))
            self.save('gift'+str(size),knock)

        def glass(t, onset, frequency, decay):
            age = np.maximum(0, t-onset)
            strike = (1-np.exp(-240*age))*np.exp(-decay*age)
            return strike*(np.sin(TAU*frequency*age) +
                           .24*np.exp(-8*age)*np.sin(TAU*frequency*2.67*age))

        t = clock(1.55)
        noise = rng.normal(size=len(t))
        rustle = np.convolve(noise,np.hanning(9)/4,mode='same')
        rustle -= np.convolve(rustle,np.ones(70)/70,mode='same')
        growth = .095*rustle*np.sin(np.pi*t/1.55)**1.4*(.75+.25*np.sin(17*t))
        for onset,frequency in ((.27,523),(.61,659),(1.0,880)):
            growth += .067*glass(t,onset,frequency,6)
        self.save('tree',growth*np.clip((1.55-t)/.06,0,1))

        t = clock(1.42)
        ornaments = np.zeros_like(t)
        for onset,frequency,decay in ((0,1397,12),(.23,1047,13),(.43,1568,11),
                                      (.74,1175,12),(.99,1319,7)):
            ornaments += .09*glass(t,onset,frequency,decay)
        self.save('decor',ornaments*np.clip((1.42-t)/.06,0,1))

    def play(self, clip):
        self.players[clip] = subprocess.Popen(
            ['/usr/bin/afplay',str(Path(self.folder.name)/(clip+'.wav'))])

    def update(self, scene):
        self.players = {k:p for k,p in self.players.items() if p.poll() is None}
        for layer in scene.started:
            if layer not in self.played_layers:
                self.play(layer)
                self.played_layers.add(layer)
        if scene.snowing:
            if not self.snow_started:
                self.play('snow_start')
                self.snow_started = True
            elif 'snow_start' not in self.players and 'snow_settle' not in self.players:
                self.play('snow_settle')
        if scene.star is not None and scene.tree >= 1 and not self.star_played:
            self.play('star')
            self.star_played = True
        if scene.ride is not self.ride:
            self.ride = scene.ride
            self.landed.clear()
            if self.ride:
                self.play('sleigh')
        if self.ride:
            for size in sorted(self.ride.landed-self.landed):
                self.play('gift'+str(size))
                self.landed.add(size)

    def reset(self):
        for p in self.players.values():
            if p.poll() is None:
                p.terminate()
            p.wait()
        self.players.clear()
        self.played_layers.clear()
        self.snow_started = False
        self.star_played = False
        self.ride = None
        self.landed.clear()

    def close(self):
        self.reset()
        self.folder.cleanup()


def main():
    import mediapipe as mp
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit('Camera unavailable. Check macOS Camera permission and close other camera apps.')
    hands = mp.solutions.hands
    states = {s: Gesture() for s in ('Left','Right')}
    scene = Scene()
    audio = WinterAudio()
    previous = time.monotonic()
    try:
        with hands.Hands(max_num_hands=2,min_detection_confidence=.65,min_tracking_confidence=.65) as tracker:
            while True:
                ok,frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame,1)
                result = tracker.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                now = time.monotonic()
                right_palm = None
                for index,(lm,label) in enumerate(zip(result.multi_hand_landmarks or [], result.multi_handedness or [])):
                    side = label.classification[0].label
                    p = np.array([(v.x*frame.shape[1],v.y*frame.shape[0]) for v in lm.landmark])
                    world = result.multi_hand_world_landmarks
                    if world:
                        coords = np.array([(v.x,v.y,v.z) for v in world[index].landmark])
                        event = states[side].read(coords,side,now)
                        if side == "Right" and states[side].spread:
                            right_palm = p[[0,5,9,13,17]].mean(axis=0) / [frame.shape[1], frame.shape[0]]
                        scene.opening(side, states[side], now)
                        scene.trigger(event,now)
                    mp.solutions.drawing_utils.draw_landmarks(frame,lm,hands.HAND_CONNECTIONS)
                    cv2.putText(frame,side,tuple(p[0].astype(int)),0,.8,(90,240,255),2)
                scene.swipe(now,right_palm)
                img = scene.draw(now,min(.12,now-previous))
                audio.update(scene)
                previous = now
                preview_w, preview_h = 225, 150
                img[12:12 + preview_h, 12:12 + preview_w] = cv2.resize(frame, (preview_w, preview_h))
                cv2.imshow('Christmas Tree',img)
                key = cv2.waitKey(1) & 255
                if key in (27,ord('q')) or cv2.getWindowProperty('Christmas Tree',cv2.WND_PROP_VISIBLE)<1:
                    break
                if key == ord('r'):
                    audio.reset()
                    scene = Scene()
                    states = {s: Gesture() for s in states}
    finally:
        audio.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
