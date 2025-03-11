// Function to apply the saved theme
let responseData = []

function applySavedTheme() {
    const body = document.body;

    // Apply dark/light mode
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode === 'true') {
        body.classList.add('dark');
    } else {
        body.classList.remove('dark');
    }

    // Apply theme color
    const savedThemeColor = localStorage.getItem('themeColor');
    if (savedThemeColor) {
        document.documentElement.style.setProperty('--primary-color', savedThemeColor);

        // Update relevant elements with the theme color
        const elementsToColor = [
            document.querySelector('header'),
            document.getElementById('sidebar'),
            ...document.querySelectorAll('button'),
            ...document.querySelectorAll('table th'),
        ];

        elementsToColor.forEach(element => {
            if (element) element.style.backgroundColor = savedThemeColor;
        });
    }
}

function getNumbers(num) {
    const n = Number(num).toFixed(2);
    if (n !== 'NaN') return n;
    return 0;
}

// database integration
async function addUser(SrNo, name_match_score, uid_match_score, final_address_match_score, overall_score, final_remarks, document_type, accepted_rejected) {
    try {
        const response = await fetch('/add_user', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ SrNo, name_match_score, uid_match_score, final_address_match_score, overall_score, final_remarks, document_type, accepted_rejected })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to add user');
        }

        return data;
    } catch (error) {
        console.error('Error adding user:', error);
        throw error;
    }
}

// Function to get users (optionally filtered by SrNo)
async function getUsers(SrNo = null) {
    try {
        const url = SrNo ? `/get_users?SrNo=${SrNo}` : '/get_users';
        
        const response = await fetch(url);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to fetch users');
        }

        return await response.json();
    } catch (error) {
        console.error('Error fetching users:', error);
        throw error;
    }
}




function getAnalyticsData(data) {
    const analyticsData = {
        '% Match': 0,
        '% Unmatched': 0,
        'Aadhaar Data': 0, // Numeric values
        'Non-Aadhaar Data': 0,
        'Total Records': 0
    };
    const totalRecords = data.length;
    for (let i = 0; i < totalRecords; i++) {
        if (data[i]['Accepted/Rejected'] === 'Accepted') {
            analyticsData['% Match'] += 1;
        } else {
            analyticsData['% Unmatched'] += 1;
        }
        if (data[i]['Document Type'] === 'Aadhaar') {
            analyticsData['Aadhaar Data'] += 1;
        } else {
            analyticsData['Non-Aadhaar Data'] += 1;
        }
    }

    // console.log({...analyticsData});
    analyticsData['% Match'] = (analyticsData['% Match'] / totalRecords) * 100;
    analyticsData['% Unmatched'] = (analyticsData['% Unmatched'] / totalRecords) * 100;
    analyticsData['Total Records'] = totalRecords;


    return analyticsData;
}

function downloadResults() {
    window.location.href = '/download';
}

// Function to render data in the table
function renderTableData(data) {

    // console.log("Render table Data: ", d);

    // const data = sessionStorage.getItem('responseData') ? JSON.parse(sessionStorage.getItem('responseData')) : d;
    const tableBody = document.querySelector('#matching-scores-table tbody');

    if (!tableBody) {
        console.error("Table body not found.");
        return;
    }

    // Clear existing rows
    tableBody.innerHTML = '';

    // Populate table rows with data
    data.forEach(item => {
        const row = document.createElement('tr');

        // Create and append Aadhaar Number cell
        const SrNo = document.createElement('td');
        SrNo.textContent = item["SrNo"] || 'N/A';
        row.appendChild(SrNo);

        // Create and append Name cell
        const nameMatchScore = document.createElement('td');
        nameMatchScore.textContent = getNumbers(item["Name Match Score"] || 'N/A');
        row.appendChild(nameMatchScore);

        // Create and append Match Score cell
        const uidMatchScore = document.createElement('td');
        uidMatchScore.textContent = getNumbers(item["UID Match Score"]);
        row.appendChild(uidMatchScore);

        // Create and append Match Score cell
        const finalAddressMatchScore = document.createElement('td');
        finalAddressMatchScore.textContent = getNumbers(item["Final Address Match Score"]);
        row.appendChild(finalAddressMatchScore);

        // Create and append Match Score cell
        const overallScore = document.createElement('td');
        overallScore.textContent = getNumbers(item["Overall Score"]);
        row.appendChild(overallScore);

        // Create and append Match Score cell
        const finalRemarks = document.createElement('td');
        finalRemarks.textContent = item["Final Remarks"];
        row.appendChild(finalRemarks);

        // Create and append Match Score cell
        const documentType = document.createElement('td');
        documentType.textContent = item["Document Type"];
        row.appendChild(documentType);

        // Create and append Match Score cell
        const acceptedRejected = document.createElement('td');
        acceptedRejected.textContent = item["Accepted/Rejected"];
        row.appendChild(acceptedRejected);

        // Append the row to the table body
        tableBody.appendChild(row);
    });
    document.getElementById('data-analytics').style.display = 'block';
}

function updateAnalytics(d) {
    // const analyticsData = sessionStorage.getItem('responseData') ? JSON.parse(sessionStorage.getItem('responseData')) : d;
    const data = getAnalyticsData(d);
    console.log(data);
    // const data = sessionStorage.getItem('responseData') ? JSON.parse(sessionStorage.getItem('responseData')) : d;
    // const analyticsData = document.querySelector('#analytics-section');

    const analyticsContainer = document.getElementById('analytics-section');

    if (!analyticsContainer) {
        console.error("Analytics container not found.");
        return;
    }

    // Clear existing analytics
    analyticsContainer.innerHTML = '';

    // Iterate through the data and create metrics
    for (const [key, value] of Object.entries(data)) {
        // Create a new metric div
        const metricDiv = document.createElement('div');
        metricDiv.className = 'metric';

        // Create label span
        const labelSpan = document.createElement('span');
        labelSpan.className = 'label';
        labelSpan.textContent = key; // key is a string

        // Create value span
        const valueSpan = document.createElement('span');
        valueSpan.style.marginLeft = "5px"
        valueSpan.className = 'value';
        valueSpan.textContent = String(value); // Convert value to string

        // Append label and value spans to the metric div
        metricDiv.appendChild(labelSpan);
        metricDiv.appendChild(valueSpan);

        // Append the metric div to the analytics container
        analyticsContainer.appendChild(metricDiv);
    }
}

// Example data
const analyticsData = {
    '% Match': '85%',
    '% Unmatched': '15%',
    'Aadhaar Data': 200, // Numeric values
    'Non-Aadhaar Data': 50,
    'Total Records': 250
};


async function handleFileUpload() {
    const uploadForm = document.getElementById("upload-form");
    const zipFileInput = document.getElementById("zip-file");
    const excelFileInput = document.getElementById("excel-file");
    const loadingIndicator = document.getElementById("loading"); // Get the loading indicator

    if (!uploadForm || !zipFileInput || !excelFileInput) {
        console.log("Upload form or inputs are missing in the DOM.");
        return;
    }

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        // Show the loading indicator when upload starts
        loadingIndicator.style.display = 'block';

        // Validate files
        const zipFile = zipFileInput.files[0];
        const excelFile = excelFileInput.files[0];

        if (!zipFile || !excelFile) {
            alert("Please upload both a ZIP file and an Excel file.");
            loadingIndicator.style.display = 'none';  // Hide loading indicator if validation fails
            return;
        }

        // Validate file types
        const zipFileName = zipFile.name.toLowerCase();
        if (!zipFileName.endsWith(".zip")) {
            alert("Invalid ZIP file. Please upload a valid .zip file.");
            loadingIndicator.style.display = 'none';  // Hide loading indicator if validation fails
            return;
        }
        if (!["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"].includes(excelFile.type)) {
            alert("Invalid Excel file. Please upload a valid .xlsx or .xls file.");
            loadingIndicator.style.display = 'none';  // Hide loading indicator if validation fails
            return;
        }

        // Create FormData and append files
        const formData = new FormData();
        formData.append("zipfile", zipFile);
        formData.append("excelfile", excelFile);

        try {
            // Send files to the server
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            const result = await response.json();
            console.log("Upload response:", result.results);
            renderTableData(result?.results);
            updateAnalytics(result?.results);
            responseData = result?.results;
            responseData.forEach(async (item) => {
                await addUser(item["SrNo"], item["Name Match Score"], item["UID Match Score"], item["Final Address Match Score"], item["Overall Score"], item["Final Remarks"], item["Document Type"], item["Accepted/Rejected"]);
            });
            // sessionStorage.setItem('responseData', JSON.stringify(result?.results));

            // Hide the loading indicator after the data is processed
            loadingIndicator.style.display = 'none';

        } catch (error) {
            console.error("Error uploading files:", error);
            alert("Error uploading files. Please try again.");
            loadingIndicator.style.display = 'none'; // Hide loading indicator if an error occurs
        }
    });
}

function formatData(data) {
    return data.map(item => {
        return {
            SrNo: item["SrNo"],
            ["Name Match Score"]: getNumbers(item.name_match_score),
            ["UID Match Score"]: getNumbers(item.uid_match_score),
            ["Final Address Match Score"]: getNumbers(item.final_address_match_score),
            ["Overall Score"]: getNumbers(item.overall_score),
            ["Final Remarks"]: item.final_remarks,
            ["Document Type"]: item.document_type,
            ["Accepted/Rejected"]: item.accepted_rejected
        };
    });
}

// Apply the saved theme and handle other initialization tasks
document.addEventListener("DOMContentLoaded", async () => {
    const dbData = await getUsers();
    console.log("dbData: ", dbData);
    const dataAnalyticsElement = document.getElementById('data-analytics');

    if (dataAnalyticsElement) {
        if (dbData.length) {
            dataAnalyticsElement.style.display = 'block';
            
            updateAnalytics(formatData(dbData) || []);

            renderTableData(formatData(dbData) || []);
        } else {
            dataAnalyticsElement.style.display = 'none';
        }
    }
    applySavedTheme();
    handleFileUpload();
    // updateCharts(responseData);
    // Sidebar Toggle
    const sidebar = document.getElementById("sidebar");
    const openSidebarBtn = document.getElementById("open-sidebar");
    const closeSidebarBtn = document.getElementById("close-sidebar");
    const dashboardContainer = document.querySelector(".dashboard-container");

    if (openSidebarBtn && closeSidebarBtn) {
        openSidebarBtn.addEventListener("click", () => {
            if (sidebar) sidebar.classList.add("open");
            if (dashboardContainer) dashboardContainer.classList.add("shift");
        });

        closeSidebarBtn.addEventListener("click", () => {
            if (sidebar) sidebar.classList.remove("open");
            if (dashboardContainer) dashboardContainer.classList.remove("shift");
        });
    }

    // Theme Toggle (Dark/Light Mode)
    const themeToggle = document.getElementById("theme-toggle");
    const body = document.body;

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const isDarkMode = body.classList.toggle("dark");
            themeToggle.innerHTML = isDarkMode
                ? '<i class="fas fa-sun"></i> Light Mode'
                : '<i class="fas fa-moon"></i> Dark Mode';

            localStorage.setItem("darkMode", isDarkMode);
        });

        const savedDarkMode = localStorage.getItem("darkMode") === "true";
        body.classList.toggle("dark", savedDarkMode);
        themeToggle.innerHTML = savedDarkMode
            ? '<i class="fas fa-sun"></i> Light Mode'
            : '<i class="fas fa-moon"></i> Dark Mode';
    }

    // Theme Customization (Color Picker)
    const themeColorInput = document.getElementById("theme-color");
    const applyThemeButton = document.getElementById("apply-theme");

    if (applyThemeButton && themeColorInput) {
        applyThemeButton.addEventListener("click", () => {
            const color = themeColorInput.value;
            document.documentElement.style.setProperty("--primary-color", color);

            const elementsToColor = [
                document.querySelector("header"),
                document.getElementById("sidebar"),
                ...document.querySelectorAll("button"),
                ...document.querySelectorAll("table th"),
            ];

            elementsToColor.forEach(element => {
                if (element) element.style.backgroundColor = color;
            });

            localStorage.setItem("themeColor", color);
        });

        const savedThemeColor = localStorage.getItem("themeColor");
        if (savedThemeColor) {
            themeColorInput.value = savedThemeColor;
        }
    }

    // Search Functionality
    const searchInput = document.getElementById("search-input");
    const tableBody = document.querySelector("#matching-scores-table tbody");

    if (searchInput && tableBody) {
        searchInput.addEventListener("input", () => {
            const searchTerm = searchInput.value.toLowerCase();
            const rows = tableBody.querySelectorAll("tr");

            rows.forEach(row => {
                const cells = row.querySelectorAll("td");
                const match = Array.from(cells).some(cell =>
                    cell.textContent.toLowerCase().includes(searchTerm)
                );
                row.style.display = match ? "" : "none";
            });
        });
    }
});