class QuadTree(object):
    """An implementation of a quad-tree.
 
    This QuadTree started life as a version of [1] but found a life of its own
    when I realised it wasn't doing what I needed. It is intended for static
    geometry, ie, items such as the landscape that don't move.
 
    This implementation inserts items at the current level if they overlap all
    4 sub-quadrants, otherwise it inserts them recursively into the one or two
    sub-quadrants that they overlap.
 
    Items being stored in the tree must possess the following attributes:
 
        left - the x coordinate of the left edge of the item's bounding box.
        top - the y coordinate of the top edge of the item's bounding box.
        right - the x coordinate of the right edge of the item's bounding box.
        bottom - the y coordinate of the bottom edge of the item's bounding box.
 
        where left &lt; right and top &lt; bottom
        
    ...and they must be hashable.
    
    Acknowledgements:
    [1] http://mu.arete.cc/pcr/syntax/quadtree/1/quadtree.py
    """
    def __init__(self, items, depth=8, bounding_rect=None):
        """Creates a quad-tree.
 
        @param items:
            A sequence of items to store in the quad-tree. Note that these
            items must possess left, top, right and bottom attributes.
            
        @param depth:
            The maximum recursion depth.
            
        @param bounding_rect:
            The bounding rectangle of all of the items in the quad-tree. For
            internal use only.
        """
        # The sub-quadrants are empty to start with.
        self.nw = self.ne = self.se = self.sw = None
        
        # If we've reached the maximum depth then insert all items into this
        # quadrant.
        depth -= 1
        if depth == 0:
            self.items = items
            return
 
        # Find this quadrant's centre.
        if bounding_rect:
            l, t, r, b = bounding_rect
        else:
            # If there isn't a bounding rect, then calculate it from the items.
            l = min(item.left for item in items)
            t = min(item.top for item in items)
            r = max(item.right for item in items)
            b = max(item.bottom for item in items)
        cx = self.cx = (l + r) * 0.5
        cy = self.cy = (t + b) * 0.5
        
        self.items = []
        nw_items = []
        ne_items = []
        se_items = []
        sw_items = []
        
        for item in items:
            # Which of the sub-quadrants does the item overlap?
            in_nw = item.left <= cx and item.top <= cy
            in_sw = item.left <= cx and item.bottom >= cy
            in_ne = item.right >= cx and item.top <= cy
            in_se = item.right >= cx and item.bottom >= cy
                
            # If it overlaps all 4 quadrants then insert it at the current
            # depth, otherwise append it to a list to be inserted under every
            # quadrant that it overlaps.
            if in_nw and in_ne and in_se and in_sw:
                self.items.append(item)
            else:
                if in_nw: nw_items.append(item)
                if in_ne: ne_items.append(item)
                if in_se: se_items.append(item)
                if in_sw: sw_items.append(item)
            
        # Create the sub-quadrants, recursively.
        if nw_items:
            self.nw = QuadTree(nw_items, depth, (l, t, cx, cy))
        if ne_items:
            self.ne = QuadTree(ne_items, depth, (cx, t, r, cy))
        if se_items:
            self.se = QuadTree(se_items, depth, (cx, cy, r, b))
        if sw_items:
            self.sw = QuadTree(sw_items, depth, (l, cy, cx, b))
 
 
    def hit(self, rect):
        """Returns the items that overlap a bounding rectangle.
 
        Returns the set of all items in the quad-tree that overlap with a
        bounding rectangle.
        
        @param rect:
            The bounding rectangle being tested against the quad-tree. This
            must possess left, top, right and bottom attributes.
        """
        def overlaps(item):
            return rect.right >= item.left and rect.left <= item.right and \
                   rect.bottom >= item.top and rect.top <= item.bottom
        
        # Find the hits at the current level.
        hits = set(item for item in self.items if overlaps(item))
        
        # Recursively check the lower quadrants.
        if self.nw and rect.left <= self.cx and rect.top <= self.cy:
            hits |= self.nw.hit(rect)
        if self.sw and rect.left <= self.cx and rect.bottom >= self.cy:
            hits |= self.sw.hit(rect)
        if self.ne and rect.right >= self.cx and rect.top <= self.cy:
            hits |= self.ne.hit(rect)
        if self.se and rect.right >= self.cx and rect.bottom >= self.cy:
            hits |= self.se.hit(rect)
 
        return hits
 
 
if __name__ == '__main__':
    import pygame
    import random
 
    # The class that we're storing in the quad-tree. It possesses the necessary
    # attributes of left, top, right and bottom, and it is hashable.
    class Item(object):
        def __init__(self, left, top, right, bottom, colour=(255, 255, 255)):
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom
            self.colour = colour
            
        def draw(self):
            x = self.left
            y = self.top
            w = self.right - x
            h = self.bottom - y
            pygame.draw.rect(screen, self.colour, pygame.Rect(x, y, w, h), 2)
 
    # Create 10,000 random items, some of which should overlap with the screen.
    colours = [
        (0, 0, 255),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (255, 255, 255)
        ]
    items = []
    for i in range(10000):
        x = random.uniform(-5000, 5000)
        y = random.uniform(-5000, 5000)
        w = 5 + random.expovariate(1.0/50)
        h = 5 + random.expovariate(1.0/50)
        colour = random.choice(colours)
        items.append(Item(x, y, x+w, y+h, colour))
        
    # Put them into a quad-tree.
    tree = QuadTree(items)
    
    WIDTH = 640
    HEIGHT = 480
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    quit = False
    while not quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                quit = True
                    
        # Use the quad-tree to restrict which items we're going to draw.
        area = pygame.Rect(0, 0, WIDTH, HEIGHT)
        visible_items = tree.hit(area)
 
        # Draw the visible items only.
        screen.fill((0, 0, 0))
        for item in visible_items:
            item.draw()
        pygame.display.flip()
    pygame.quit()
 
    print "Total items:", len(items)
    print "Visible items:", len(visible_items)
