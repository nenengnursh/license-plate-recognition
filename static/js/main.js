/*=============== SHOW MENU ===============*/
const showMenu = (toggleId, navId) => {
  const toggle = document.getElementById(toggleId),
    nav = document.getElementById(navId);

  toggle.addEventListener("click", () => {
    // Add show-menu class to nav menu
    nav.classList.toggle("show-menu");

    // Add show-icon to show and hide the menu icon
    toggle.classList.toggle("show-icon");
  });
};

showMenu("nav-toggle", "nav-menu");

function updateFileName() {
  const input = document.getElementById("input-image");
  const filename = document.getElementById("filename-image");
  const btnLabel = document.querySelector(".btn-label-image");

  filename.textContent = input.files[0]
    ? input.files[0].name
    : "No image selected";

  if (input.files.length > 0) {
    btnLabel.classList.add("active");
  } else {
    btnLabel.classList.remove("active");
  }
}

function updateFolderName() {
  const input = document.getElementById("input-folder");
  const filename = document.getElementById("filename-folder");
  const btnLabel = document.querySelector(".btn-label-folder");

  if (input.files.length > 0) {
    const folderName = input.files[0].webkitRelativePath.split("/")[0];
    filename.textContent = "folder " + folderName;
    btnLabel.classList.add("active");
  } else {
    filename.textContent = "No folder selected";
    btnLabel.classList.remove("active");
  }
}

function updateFileNameAndShowDatatrainGrid() {
  updateFileName(); // Memanggil fungsi updateFileName()
  showDatatrainGrid(); // Memanggil fungsi showDatatrainGrid()
}

function updateFileName() {
  const input = document.getElementById("input-image");
  const filename = document.getElementById("filename-image");
  const btnLabel = document.querySelector(".btn-label-image");

  filename.textContent = input.files[0]
    ? input.files[0].name
    : "No image selected";

  if (input.files.length > 0) {
    btnLabel.classList.add("active");
  } else {
    btnLabel.classList.remove("active");
  }
}

function showDatatrainGrid() {
  var trainingGrid = document.querySelector(".datatrain-grid");
  var totalImage = document.querySelector(".jumlah-gambar");
  trainingGrid.style.display = "flex";
  totalImage.style.display = "flex";
}

function updateFileNameAndShowImage() {
  updateFileName();
  showImage();
}

function showImage() {
  var trainingGrid = document.querySelector(".gambar");
  trainingGrid.style.display = "grid";
}

function progressDisplay() {
  var trainingButton = document.querySelector(".submit");
  var progressBar = document.querySelector(".processing");
  trainingButton.style.display = "none";
  progressBar.style.display = "flex";
}
