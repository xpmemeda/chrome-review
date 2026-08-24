export TMPDIR=${HOME}/.tmp

if [ ! -d ${TMPDIR} ]; then
    echo "Creating temp directory: ${TMPDIR}"
    mkdir -p ${TMPDIR}
fi