#[cfg(test)]
mod test;

/// What a kernel needs of a multiprocessor of the device, which limits how many of its threads
/// the device runs at once, and so how well it hides the latency of its memory and arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resources {
    /// The registers of each thread, of at most 255, as the driver compiled the kernel for the
    /// device.
    pub registers: u32,
    /// The bytes of local memory of each thread, which is the memory of the device, and slow.
    pub local_bytes: u32,
    /// The threads in a block that the figures below are for.
    pub threads_per_block: u32,
    /// How many of those blocks fit on a multiprocessor at once.
    pub blocks_per_multiprocessor: u32,
    /// The most threads that a multiprocessor has at once.
    pub threads_per_multiprocessor: u32,
}

impl Resources {
    /// The fraction of the threads of a multiprocessor that the kernel has at once.
    pub fn occupancy(&self) -> f64 {
        f64::from(self.blocks_per_multiprocessor * self.threads_per_block)
            / f64::from(self.threads_per_multiprocessor.max(1))
    }
}
