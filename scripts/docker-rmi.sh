NAME=""

while getopts "n:" opt; do
  case $opt in
    n)
      NAME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ $NAME == "" ]]; then
  echo "Usage: $0 -n <image name>"
  exit 1
fi

docker images | grep $NAME | awk '{print $1":"$2}' | xargs -r docker rmi
