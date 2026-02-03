#include "ryzenai/driver_check.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <comdef.h>
#include <Wbemidl.h>

#pragma comment(lib, "wbemuuid.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#endif

namespace ryzenai {

const std::string RYZENAI_SERVER_MINIMUM_DRIVER = "32.0.203.280";
const std::string DRIVER_INSTALL_URL = "https://lemonade-server.ai/driver_install";

#ifdef _WIN32

// Minimal WMI Helper classes to avoid full dependency
class WMIConnection {
public:
    WMIConnection() {
        HRESULT hres = CoInitializeEx(0, COINIT_MULTITHREADED);
        if (FAILED(hres) && hres != RPC_E_CHANGED_MODE) return;
        
        hres = CoInitializeSecurity(NULL, -1, NULL, NULL, RPC_C_AUTHN_LEVEL_DEFAULT, 
                                  RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE, NULL);
        
        hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, 
                              IID_IWbemLocator, (LPVOID*)&pLoc_);
        if (FAILED(hres)) return;
        
        hres = pLoc_->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), NULL, NULL, 0, NULL, 0, 0, &pSvc_);
        if (FAILED(hres)) { pLoc_->Release(); pLoc_ = nullptr; return; }
        
        hres = CoSetProxyBlanket(pSvc_, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL, 
                               RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE);
        if (FAILED(hres)) { pSvc_->Release(); pSvc_ = nullptr; pLoc_->Release(); pLoc_ = nullptr; }
    }

    ~WMIConnection() {
        if (pSvc_) pSvc_->Release();
        if (pLoc_) pLoc_->Release();
        // CoUninitialize(); // Usually better not to call this if we didn't init successfully or if app uses COM elsewhere
    }

    bool is_valid() const { return pSvc_ != nullptr; }

    bool query(const std::wstring& wql, std::string& result_version) {
        if (!pSvc_) return false;
        
        IEnumWbemClassObject* pEnumerator = nullptr;
        HRESULT hres = pSvc_->ExecQuery(bstr_t("WQL"), bstr_t(wql.c_str()), 
                                      WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, NULL, &pEnumerator);
        if (FAILED(hres)) return false;

        IWbemClassObject* pclsObj = nullptr;
        ULONG uReturn = 0;
        bool found = false;

        while (pEnumerator) {
            HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
            if (0 == uReturn) break;

            VARIANT vtProp;
            VariantInit(&vtProp);
            hr = pclsObj->Get(L"DriverVersion", 0, &vtProp, 0, 0);
            
            if (SUCCEEDED(hr) && vtProp.vt == VT_BSTR && vtProp.bstrVal != nullptr) {
                std::wstring wstr(vtProp.bstrVal, SysStringLen(vtProp.bstrVal));
                int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
                std::string strTo(size_needed, 0);
                WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
                result_version = strTo;
                found = true;
            }
            VariantClear(&vtProp);
            pclsObj->Release();
            if (found) break;
        }
        pEnumerator->Release();
        return found;
    }

private:
    IWbemLocator* pLoc_ = nullptr;
    IWbemServices* pSvc_ = nullptr;
};

/**
 * @brief Retrieves the NPU driver version string from the Windows system.
 *
 * This function uses WMI to query the driver version of the NPU Compute Accelerator Device.
 *
 * @return std::string The driver version string, or empty string if not found or WMI fails.
 */
std::string GetNPUDriverVersion() {
    WMIConnection wmi;
    if (!wmi.is_valid()) return "";

    std::string version;
    // Query for "NPU Compute Accelerator Device"
    // Need to escape special characters in query if any, but specific name is simple
    std::wstring query = L"SELECT DriverVersion FROM Win32_PnPSignedDriver WHERE DeviceName LIKE '%NPU Compute Accelerator Device%'";
    
    if (wmi.query(query, version)) {
        return version;
    }
    return "";
}

/**
 * @brief Opens the default web browser to the specified URL.
 *
 * This function uses ShellExecute to launch the default browser with the given URL.
 *
 * @param url The URL to open in the browser.
 */
void OpenBrowser(const std::string& url) {
    ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
}

#else

#if defined(__APPLE__) || defined(__linux__)
std::string GetNPUDriverVersion() { return ""; }  // Return empty on macOS and Linux to skip NPU version check
#else
std::string GetNPUDriverVersion() { return "0.0.0.0"; }
#endif

void OpenBrowser(const std::string& url) {
    std::cout << "Please visit: " << url << std::endl;
}

#endif

/**
 * @brief Compares two version strings to determine if the first is less than the second.
 *
 * This function parses version strings in dotted notation (e.g., "1.2.3.4") and compares
 * them component by component to determine ordering.
 *
 * @param v1 First version string to compare.
 * @param v2 Second version string to compare.
 * @return bool True if v1 is less than v2, false otherwise.
 */
bool IsVersionLessThan(const std::string& v1, const std::string& v2) {
    auto parse_version = [](const std::string& v) -> std::vector<int> {
        std::vector<int> parts;
        std::stringstream ss(v);
        std::string part;
        while (std::getline(ss, part, '.')) {
            try {
                parts.push_back(std::stoi(part));
            } catch (...) {
                parts.push_back(0);
            }
        }
        while (parts.size() < 4) parts.push_back(0); // Ensure 4 parts
        return parts;
    };

    std::vector<int> parts1 = parse_version(v1);
    std::vector<int> parts2 = parse_version(v2);

    size_t max_size = (std::max)(parts1.size(), parts2.size());
    parts1.resize(max_size, 0);
    parts2.resize(max_size, 0);

    for (size_t i = 0; i < max_size; ++i) {
        if (parts1[i] < parts2[i]) return true;
        if (parts1[i] > parts2[i]) return false;
    }
    return false;
}

/**
 * @brief Checks if the installed NPU driver version meets the minimum requirements.
 *
 * This function retrieves the current NPU driver version and compares it against
 * the minimum required version for Ryzen AI server. If the version is too old,
 * it displays an error message and opens a browser to the driver download page.
 *
 * @return bool True if the driver version is sufficient or undetectable, false if too old.
 */
bool CheckNPUDriverVersion() {
    std::string version = GetNPUDriverVersion();
    std::cout << "[DEBUG] GetNPUDriverVersion returned: '" << version << "'" << std::endl;

    if (version.empty()) {
        std::cout << "[Server] NPU Driver Version: Unknown (Could not detect)" << std::endl;
#ifdef __APPLE__
        // On macOS, Ryzen AI drivers don't exist, so MLX is the only backend
        // Allow startup to proceed for MLX models
        std::cout << "[Server] macOS detected - allowing startup for MLX backend" << std::endl;
        return true;
#endif
        return true; // Assume OK if we can't detect, to not block users with weird setups
    }
    
    std::cout << "[Server] NPU Driver Version: " << version << std::endl;
    
    if (IsVersionLessThan(version, RYZENAI_SERVER_MINIMUM_DRIVER)) {
        std::cerr << "\n===============================================================" << std::endl;
        std::cerr << "ERROR: NPU Driver Version is too old!" << std::endl;
        std::cerr << "Current: " << version << std::endl;
        std::cerr << "Minimum: " << RYZENAI_SERVER_MINIMUM_DRIVER << std::endl;
        std::cerr << "Please update your NPU driver at: " << DRIVER_INSTALL_URL << std::endl;
        std::cerr << "===============================================================\n" << std::endl;
        
        OpenBrowser(DRIVER_INSTALL_URL);
        return false;
    }

    return true;
}

} // namespace ryzenai

