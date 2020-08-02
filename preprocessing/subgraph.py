class SubgraphLoader():
    def __init__(self, 
                 inputShape, 
                 sampleShape, 
                 keys, 
                 keyToPath,
                 labelFile):
        self.inputShape = inputShape
        self.sampleShape = sampleShape
        self.keys = keys
        self.keyToPath = keyToPath
        self.particles = self._parseParticles(labelFile)
    
    def getMicrograph(self, key):
        """ Load micrograph. """
        with mrcfile.open(self.keyToPath(key), permissive=True) as mrc:
            data = mrc.data
        return data
    
    def boxContains(point, sampleId):
        dimH, dimW = self.sampleShape 
        x, y = point
        cx, cy = sampleId
        return (cx*dimW < x < cx*dimW + dimW) and (cy*dimH < y < cy*dimH + dimH)
    
    def _generateSubgraph(self, key):
        """ Generate the subimages for a given micrograph. """
        retDict = {}
        h, w = self.inputShape
        dimH, dimW = self.sampleShape
        data = self.getMicrograph(key)
        for idxh in range(int(h/dimH)):
            for idxw in range(int(w/dimW)):
                retDict[(idxh,idxw)] = data[idxh*dimH:idxh*dimH+dimH, 
                                                  idxw*dimW:idxw*dimW+dimW]
        return retDict

    def _parseParticles(self, file):
        """ 
        Read in the particles for all micrographs. 
        This will need to be edited if you change your key types. 
        """
        with open(file, "r") as f:
            particles = f.readlines()

        particleData = [particle.split()[0:3] for particle in particles[17:-1]]
        particleDict = {}
        for x in particleData:
            key = int(x[0][18:22])
            value = tuple(map(float, x[1:]))
            particleDict.setdefault(key, []).append(value)
        return particleDict

    def getSubgraphAnnotation(self, shift = True):
        """ 
        Searches through the particle list a dictionary which maps
                
        (micrographKey, subgraphKey) -> [particles in subgraph]

        subgraphKey - is the x,y position of a subgraph within the grid formed by the subgraphs over the micrograph.
        shift specifies if you want the absolute or relative posiotions.
        """
        subDict = {}
        h, w = self.inputShape
        dimH, dimW = self.sampleShape
        for micrograph in self.keys:
            for idxh in range(int(h/dimH)):
                for idxw in range(int(w/dimW)):
                    subgraph_particles = np.array(
                        list(
                            filter(
                                lambda x : self.boxContains(x, (idxh, idxw), dimH, dimW), 
                                self.particles[micrograph]
                            )
                        )
                    )
                    try:
                        subDict[(micrograph, idxh, idxw)] = subgraph_particles - np.array([idxh*dimH, idxw*dimW])
                    except:
                        continue
        return subDict
    
    def getSubgraphs(self):
        subDict = {}
        for micrograph in self.keys:
            subgraphs = self._generateSubgraph(micrograph)
            for k,v in subgraphs.items():
                subDict[(micrograph, *k)] = v
        return subDict