#[cfg(all(test, feature = "std"))]
mod test;

/// How a computation is spread over threads.
///
/// The choice never changes the results, which are identical in either case, and only the run
/// is affected: the threads and their storage are created when the run starts and dropped
/// when it ends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Execution {
    /// On the calling thread, with nothing created for the run.
    #[default]
    Serial,
    /// On the given number of threads, which live for the run. Any number, including one,
    /// takes the parallel pathway, which [`Serial`](Self::Serial) avoids, and zero is as many
    /// threads as the machine offers.
    Threads(usize),
}

#[cfg(feature = "std")]
impl Execution {
    /// The number of threads the computation is spread over, at least one: one for
    /// [`Serial`](Self::Serial), and for [`Threads`](Self::Threads) the number given, or the
    /// number the machine offers if it is zero.
    pub fn threads(self) -> usize {
        match self {
            Self::Serial => 1,
            Self::Threads(0) => std::thread::available_parallelism().map_or(1, usize::from),
            Self::Threads(threads) => threads,
        }
    }
}
