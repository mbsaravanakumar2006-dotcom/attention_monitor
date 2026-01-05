class PersonTracker:
    def __init__(self):
        self.next_object_id = 0
        self.objects = {} # ID -> Centroid
        
    def update(self, rects):
        # Associate new rects with existing IDs
        # Return list of objects with IDs
        return self.objects
